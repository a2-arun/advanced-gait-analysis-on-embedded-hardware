"""Model slot: receive, inspect and explicitly activate a finalized deployment package.

A package is a directory with `contract.json` (nano-deployment-contract-v1), `manifest.json`,
`weights.npz` and, from the research owner, `approval.json`; the Nano adds `parity_report.json`.
Readiness is reported as separate stages (never one boolean) and activation requires all of
them. Nothing here selects a model, installs a threshold or falls back to another package.
"""
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import load_arrays
from .backend import POSE_SETTINGS, backend_for
from .contract import identifier, sha256_file
from .deployment import UnsupportedModel, gaitpt_grew_contract, semantic_id, validate_contract
from .registry import validate_datetime

STAGES = ('discovered', 'structurally_valid', 'runtime_compatible', 'parity_tested',
          'approved', 'active')
ACTIVE_FILE = 'active.json'
ACTIVE_SCHEMA = 'nano-active-model-v1'
APPROVAL_SCHEMA = 'nano-model-approval-v1'
PARITY_SCHEMA = 'nano-model-parity-v1'
INVENTORY_SCHEMA = 'nano-model-inventory-v1'
BASELINE_ID = 'development-baseline'
NANO_POSE_VERSION = '0.8.5'  # the only pose stack the Nano pose graph check accepts
PACKAGE_FILES = ('contract.json', 'manifest.json', 'weights.npz', 'approval.json',
                 'parity_report.json')
ORDER = {'PASS': 0, 'WARNING': 1, 'NOT_TESTED': 2, 'BLOCKED': 3, 'FAIL': 4}


class ModelNotDeployed(RuntimeError):
    """No usable active model: a non-decision, never a fallback."""


class ModelIncompatible(ValueError):
    """The package or active model does not satisfy a required activation stage."""


def worst(statuses):
    return max(statuses, key=ORDER.__getitem__) if statuses else 'PASS'


def stage(status, detail=''):
    return dict(status=status, detail=detail)


def on_nano(facts):
    hardware = (facts or {}).get('hardware', '')
    return 'Jetson Nano' in hardware and 'Orin' not in hardware


def read_json(path):
    return json.loads(Path(path).read_text())


def expected_contract(contract):
    """What this Nano build implements for the declared family; raises UnsupportedModel."""
    backend = backend_for(contract)
    if backend.family != 'gaitpt':
        raise UnsupportedModel('No contract builder for family ' + backend.family)
    pose = contract['pose']
    return gaitpt_grew_contract(dict(weights_sha256=contract['model']['weights_sha256']),
                                dict(version=pose['version'], settings=POSE_SETTINGS,
                                     asset_sha256=pose['asset_sha256']))


def structural_stage(directory):
    for name in ('contract.json', 'manifest.json', 'weights.npz'):
        if not (directory / name).is_file():
            return None, stage('FAIL', 'missing ' + name)
    try:
        contract = validate_contract(read_json(directory / 'contract.json'))
        manifest = read_json(directory / 'manifest.json')
        digest = sha256_file(directory / 'weights.npz')
        if digest != contract['model']['weights_sha256'] or digest != manifest.get('weights_sha256'):
            return contract, stage('FAIL', 'weights SHA-256 differs from contract/manifest')
        load_arrays(directory / 'weights.npz')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return None, stage('FAIL', 'invalid package: ' + str(exc))
    return contract, stage('PASS', 'contract valid; weights hash and pickle-free tensors verified')


def requirement_checks(requirements, facts):
    """Compare declared runtime_requirements with the Nano's facts; unknown keys are untestable."""
    results = []
    packages = facts.get('packages', {})
    rules = dict(python=lambda want: str(facts.get('python', '')).startswith(want),
                 torch=lambda want: str(packages.get('torch', '')).startswith(want),
                 cuda=lambda want: facts.get('cuda') == want,
                 mediapipe=lambda want: packages.get('mediapipe') == want)
    for key, want in sorted(requirements.items()):
        if key not in rules:
            results.append(('NOT_TESTED', 'requirement ' + key + ' cannot be verified automatically'))
        elif not rules[key](str(want)):
            results.append(('FAIL', 'requirement {}={} not met'.format(key, want)))
    return results


def runtime_stage(contract, directory, facts, source_dir):
    checks = []
    try:
        backend = backend_for(contract)
        expected = expected_contract(contract)
    except UnsupportedModel as exc:
        return stage('FAIL', 'unsupported model family: ' + str(exc))
    differing = [name for name in ('model', 'pose', 'preprocessing', 'embedding', 'scoring', 'numeric')
                 if contract[name] != expected[name]]
    if differing:
        return stage('FAIL', 'differs from the Nano implementation: ' + ', '.join(differing))
    if contract['pose']['version'] != NANO_POSE_VERSION:
        return stage('FAIL', 'pose version {} is not the Nano stack {}'.format(
            contract['pose']['version'], NANO_POSE_VERSION))
    try:
        backend.validate_artifacts(source_dir, str(directory))
    except Exception as exc:
        return stage('FAIL', 'artifact/source validation failed: ' + str(exc))
    if not on_nano(facts):
        return stage('NOT_TESTED', 'static contract matches; runtime compatibility needs the Nano')
    checks.extend(requirement_checks(contract.get('runtime_requirements', {}), facts))
    status = worst([c[0] for c in checks])
    return stage(status, '; '.join(c[1] for c in checks) or 'contract matches the Nano implementation')


def parity_stage(directory, sid, digest):
    path = directory / 'parity_report.json'
    if not path.is_file():
        return stage('NOT_TESTED', 'no parity_report.json recorded for this package')
    try:
        record = read_json(path)
        report = record['report']
        if (record['schema'] != PARITY_SCHEMA or record['semantic_id'] != sid
                or record['weights_sha256'] != digest):
            return stage('FAIL', 'parity report belongs to a different package')
        if report.get('passed') is not True or report.get('not_run'):
            return stage('FAIL', 'parity report did not pass every level')
        hardware = record['environment'].get('hardware', '')
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        return stage('FAIL', 'unreadable parity report: ' + str(exc))
    if not on_nano(dict(hardware=hardware)):
        return stage('WARNING', 'parity was measured off the Nano; rerun on the original Nano')
    return stage('PASS', 'all parity levels passed on the Nano')


def approval_stage(directory, contract, sid, digest):
    if contract.get('provenance', {}).get('development_only'):
        return stage('FAIL', 'development-only package cannot be approved')
    path = directory / 'approval.json'
    if not path.is_file():
        return stage('NOT_TESTED', 'no approval.json from the research owner')
    try:
        record = read_json(path)
        if (record['schema'] != APPROVAL_SCHEMA or record['semantic_id'] != sid
                or record['weights_sha256'] != digest):
            return stage('FAIL', 'approval is for a different model/contract')
        if not record['approved_by'] or not record['research_reference']:
            return stage('FAIL', 'approval lacks approver or research reference')
        if validate_datetime(record['approved_at']).utcoffset() is None:
            return stage('FAIL', 'approval time lacks a UTC offset')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return stage('FAIL', 'invalid approval: ' + str(exc))
    return stage('PASS', 'approved by ' + record['approved_by'])


def read_active(slot):
    try:
        record = read_json(Path(slot) / ACTIVE_FILE)
        if record['schema'] == ACTIVE_SCHEMA and isinstance(record['package_id'], str):
            return record
    except (ValueError, OSError, KeyError, TypeError):
        pass
    return None


def inspect_package(path, facts, source_dir):
    directory = Path(path)
    report = dict(package_id=directory.name, family=None, variant=None, semantic_id=None,
                  development_only=False, stages={}, activatable=False, blockers=[])
    found = directory.is_dir()
    report['stages']['discovered'] = stage('PASS' if found else 'FAIL',
                                           'package directory present' if found else 'not found')
    contract, structure = (structural_stage(directory) if found else (None, stage('BLOCKED')))
    report['stages']['structurally_valid'] = structure
    if contract is not None:
        report.update(family=contract['model']['family'], variant=contract['model']['variant'],
                      semantic_id=semantic_id(contract),
                      development_only=bool(contract.get('provenance', {}).get('development_only')))
    if structure['status'] == 'PASS':
        digest = contract['model']['weights_sha256']
        report['stages']['runtime_compatible'] = runtime_stage(contract, directory, facts, source_dir)
        report['stages']['parity_tested'] = parity_stage(directory, report['semantic_id'], digest)
        report['stages']['approved'] = approval_stage(directory, contract, report['semantic_id'], digest)
    else:
        for name in ('runtime_compatible', 'parity_tested', 'approved'):
            report['stages'][name] = stage('BLOCKED', 'package is not structurally valid')
    gates = [report['stages'][name]['status'] for name in STAGES[:5]]
    report['activatable'] = all(g == 'PASS' for g in gates) and not report['development_only']
    report['blockers'] = ['{}: {}'.format(n, report['stages'][n]['detail'])
                          for n in STAGES[:5] if report['stages'][n]['status'] != 'PASS']
    return with_active(report, directory.parent)


def with_active(report, slot):
    record = read_active(slot)
    is_active = bool(record) and record['package_id'] == report['package_id'] \
        and record['semantic_id'] == report['semantic_id'] and report['activatable']
    if is_active:
        report['stages']['active'] = stage('PASS', 'active model')
    elif report['activatable']:
        report['stages']['active'] = stage('NOT_TESTED', 'valid but not activated')
    else:
        report['stages']['active'] = stage('BLOCKED', 'earlier stages are not all PASS')
    return report


def baseline_entry(bundle):
    present = bool(bundle) and (Path(bundle) / 'manifest.json').is_file() \
        and (Path(bundle) / 'weights.npz').is_file()
    note = ('provisional frozen GaitPT bundle (development only; not the final research model)')
    return dict(package_id=BASELINE_ID, family='gaitpt', variant='grew-official', semantic_id=None,
                development_only=True, activatable=False, blockers=[note], stages=dict(
                    discovered=stage('PASS' if present else 'FAIL', note if present else 'bundle absent'),
                    structurally_valid=stage('NOT_TESTED', 'checked by `gait.py check` on the Nano'),
                    runtime_compatible=stage('NOT_TESTED', 'checked by `gait.py check` on the Nano'),
                    parity_tested=stage('NOT_TESTED', 'not part of the final-model gate'),
                    approved=stage('FAIL', 'development baseline is never an approved final model'),
                    active=stage('FAIL', 'never activated; use --development-baseline explicitly')))


def inventory(slot, facts, source_dir, baseline_bundle=None):
    slot = Path(slot)
    names = sorted(p.name for p in slot.iterdir() if p.is_dir()) if slot.is_dir() else []
    packages = [inspect_package(slot / n, facts, source_dir) for n in names]
    active = [p['package_id'] for p in packages if p['stages']['active']['status'] == 'PASS']
    return dict(schema=INVENTORY_SCHEMA, state='ACTIVE' if active else 'MODEL_NOT_DEPLOYED',
                active=active[0] if active else None, packages=packages,
                development_baseline=baseline_entry(baseline_bundle) if baseline_bundle else None)


def package_path(slot, package_id):
    identifier(package_id)
    return Path(slot) / package_id


def activate(slot, package_id, facts, source_dir):
    """Point the slot at an already installed, fully qualified package. Never installs."""
    try:
        directory = package_path(slot, package_id)
    except ValueError as exc:
        raise ModelIncompatible(str(exc))
    report = inspect_package(directory, facts, source_dir)
    if not report['activatable']:
        raise ModelIncompatible('Package is not activatable: ' + '; '.join(report['blockers'] or [
            'development-only']))
    record = dict(schema=ACTIVE_SCHEMA, package_id=package_id, semantic_id=report['semantic_id'],
                  activated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    Path(slot).mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=str(slot), prefix='.active-')
    with os.fdopen(fd, 'w') as stream:
        json.dump(record, stream)
    os.chmod(temporary, 0o600)
    os.replace(temporary, str(Path(slot) / ACTIVE_FILE))
    return record


def deactivate(slot):
    path = Path(slot) / ACTIVE_FILE
    if path.exists():
        path.unlink()


def active_package(slot, facts, source_dir):
    """The one explicitly activated package, re-qualified now; anything else is ModelNotDeployed."""
    record = read_active(slot)
    if record is None:
        raise ModelNotDeployed('No model has been activated')
    try:
        directory = package_path(slot, record['package_id'])
    except ValueError:
        raise ModelNotDeployed('Active model pointer is invalid')
    report = inspect_package(directory, facts, source_dir)
    if not report['activatable'] or report['semantic_id'] != record['semantic_id']:
        raise ModelNotDeployed('Active model no longer satisfies its activation requirements')
    contract = read_json(directory / 'contract.json')
    return dict(package_id=record['package_id'], path=str(directory), contract=contract,
                semantic_id=report['semantic_id'])


def install_package(source, slot):
    """Copy a structurally valid package into the slot. Does not activate or overwrite."""
    source, slot = Path(source), Path(slot)
    name = identifier(source.name)
    if not source.is_dir():
        raise ValueError('Package must be a directory')
    extras = sorted(p.name for p in source.iterdir() if p.name not in PACKAGE_FILES)
    if extras:
        raise ValueError('Unexpected files in package: ' + ', '.join(extras))
    contract, structure = structural_stage(source)
    if structure['status'] != 'PASS':
        raise ValueError('Package is not structurally valid: ' + structure['detail'])
    target = slot / name
    if target.exists():
        raise ValueError('A package with this name is already installed')
    slot.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(dir=str(slot), prefix='.install-'))
    try:
        for item in source.iterdir():
            shutil.copyfile(str(item), str(staging / item.name))
            os.chmod(str(staging / item.name), 0o600)
        os.rename(str(staging), str(target))
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
    return target


def record_parity(directory, fixture_path, encode, facts):
    """Run the research parity fixture against this package and store the evidence in it.
    `encode(crops) -> embedding` is the package's model (None leaves embedding levels NOT_RUN)."""
    from .parity import ParityError, run_parity
    directory = Path(directory)
    contract, structure = structural_stage(directory)
    if structure['status'] != 'PASS':
        raise ValueError('Package is not structurally valid: ' + structure['detail'])
    fixture = read_json(fixture_path)
    try:
        report = run_parity(fixture, contract, encode)
    except ParityError as exc:
        raise ValueError(str(exc))
    packages = (facts or {}).get('packages', {})
    record = dict(schema=PARITY_SCHEMA, semantic_id=semantic_id(contract),
                  weights_sha256=contract['model']['weights_sha256'],
                  fixture_sha256=sha256_file(fixture_path),
                  environment=dict(hardware=(facts or {}).get('hardware', ''),
                                   machine=(facts or {}).get('machine'),
                                   python=(facts or {}).get('python'), torch=packages.get('torch'),
                                   cuda=(facts or {}).get('cuda')),
                  recorded_at=datetime.now(timezone.utc).isoformat(timespec='seconds'), report=report)
    fd, temporary = tempfile.mkstemp(dir=str(directory), prefix='.parity-')
    with os.fdopen(fd, 'w') as stream:
        json.dump(record, stream, indent=2)
    os.chmod(temporary, 0o600)
    os.replace(temporary, str(directory / 'parity_report.json'))
    return record

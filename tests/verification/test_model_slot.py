import json
import shutil
from pathlib import Path

import pytest
from lab_helpers import host_facts, make_package, nano_facts

from src.verification import model_slot
from src.verification.backend import LocalGaitPT
from src.verification.model_slot import (ModelIncompatible, ModelNotDeployed, STAGES, activate,
                                         active_package, deactivate, inspect_package, install_package,
                                         inventory, record_parity)


@pytest.fixture(autouse=True)
def pinned_source_is_present(monkeypatch):
    # The pinned official GaitPT clone is not available in tests; hash checks are exercised elsewhere.
    monkeypatch.setattr(LocalGaitPT, 'validate_artifacts', staticmethod(lambda source, bundle: {}))


def stages(report):
    return {name: report['stages'][name]['status'] for name in STAGES}


def test_stages_are_distinct_not_one_boolean(tmp_path):
    report = inspect_package(make_package(tmp_path), nano_facts(), source_dir='src')
    assert STAGES == ('discovered', 'structurally_valid', 'runtime_compatible', 'parity_tested',
                      'approved', 'active')
    assert stages(report) == dict(discovered='PASS', structurally_valid='PASS',
                                  runtime_compatible='PASS', parity_tested='PASS',
                                  approved='PASS', active='NOT_TESTED')
    assert report['activatable'] is True and report['development_only'] is False


def test_empty_slot_reports_model_not_deployed(tmp_path):
    state = inventory(tmp_path / 'slot', nano_facts(), source_dir='src')
    assert state['state'] == 'MODEL_NOT_DEPLOYED' and state['active'] is None
    assert state['packages'] == []
    with pytest.raises(ModelNotDeployed):
        active_package(tmp_path / 'slot', nano_facts(), source_dir='src')


def test_missing_parity_or_approval_blocks_activation_but_not_other_stages(tmp_path):
    no_parity = inspect_package(make_package(tmp_path, 'a', parity=False), nano_facts(), 'src')
    assert stages(no_parity)['parity_tested'] == 'NOT_TESTED'
    assert stages(no_parity)['approved'] == 'PASS' and not no_parity['activatable']
    no_approval = inspect_package(make_package(tmp_path, 'b', approve=False), nano_facts(), 'src')
    assert stages(no_approval)['approved'] == 'NOT_TESTED' and not no_approval['activatable']
    for name in ('a', 'b'):
        with pytest.raises(ModelIncompatible):
            activate(tmp_path, name, nano_facts(), 'src')
    assert not (tmp_path / 'active.json').exists()


def test_tampered_weights_fail_structure_and_block_later_stages(tmp_path):
    package = make_package(tmp_path)
    with open(str(package / 'weights.npz'), 'ab') as stream:
        stream.write(b'x')
    result = stages(inspect_package(package, nano_facts(), 'src'))
    assert result['structurally_valid'] == 'FAIL'
    assert result['runtime_compatible'] == result['parity_tested'] == 'BLOCKED'


@pytest.mark.parametrize('edit,needle', [
    (lambda c: c['model'].update(family='gpgait'), 'family'),
    (lambda c: c['pose'].update(estimator='rtmpose-m'), 'pose'),
    (lambda c: c['preprocessing'].update(joint_order=list(range(18))), 'preprocessing'),
    (lambda c: c['embedding'].update(dim=4864), 'embedding'),
])
def test_unsupported_family_pose_or_preprocessing_fails_runtime_stage(tmp_path, edit, needle):
    report = inspect_package(make_package(tmp_path, edit=edit), nano_facts(), 'src')
    assert report['stages']['runtime_compatible']['status'] == 'FAIL'
    assert needle in report['stages']['runtime_compatible']['detail']
    assert not report['activatable']


def test_declared_runtime_requirements_are_checked_on_the_nano_only(tmp_path):
    def require(c):
        c['runtime_requirements'] = dict(python='3.6', torch='1.10', cuda='10.2')
    ok = inspect_package(make_package(tmp_path, 'ok', edit=require), nano_facts(), 'src')
    assert stages(ok)['runtime_compatible'] == 'PASS'
    unmet = inspect_package(make_package(tmp_path, 'bad', edit=lambda c: c.update(
        runtime_requirements=dict(torch='2.1'))), nano_facts(), 'src')
    assert stages(unmet)['runtime_compatible'] == 'FAIL'
    workstation = inspect_package(make_package(tmp_path, 'host', edit=require), host_facts(), 'src')
    assert stages(workstation)['runtime_compatible'] == 'NOT_TESTED'
    assert not workstation['activatable']


def test_missing_pinned_source_is_a_runtime_failure(tmp_path, monkeypatch):
    def missing(source, bundle):
        raise FileNotFoundError('gaitpt.py')
    monkeypatch.setattr(LocalGaitPT, 'validate_artifacts', staticmethod(missing))
    report = inspect_package(make_package(tmp_path), nano_facts(), 'src')
    assert stages(report)['runtime_compatible'] == 'FAIL'


def test_parity_must_match_package_and_run_on_the_nano(tmp_path):
    wrong = make_package(tmp_path, 'wrong')
    report = json.loads((wrong / 'parity_report.json').read_text())
    report['semantic_id'] = '0' * 64
    (wrong / 'parity_report.json').write_text(json.dumps(report))
    assert stages(inspect_package(wrong, nano_facts(), 'src'))['parity_tested'] == 'FAIL'
    failed = make_package(tmp_path, 'failed')
    report = json.loads((failed / 'parity_report.json').read_text())
    report['report']['passed'] = False
    (failed / 'parity_report.json').write_text(json.dumps(report))
    assert stages(inspect_package(failed, nano_facts(), 'src'))['parity_tested'] == 'FAIL'
    partial = make_package(tmp_path, 'partial')
    report = json.loads((partial / 'parity_report.json').read_text())
    report['report']['not_run'] = ['walk_embedding']
    (partial / 'parity_report.json').write_text(json.dumps(report))
    assert stages(inspect_package(partial, nano_facts(), 'src'))['parity_tested'] == 'FAIL'
    workstation = make_package(tmp_path, 'ws', parity_hardware='x86_64 laptop')
    result = inspect_package(workstation, nano_facts(), 'src')
    assert stages(result)['parity_tested'] == 'WARNING' and not result['activatable']


def test_approval_must_bind_this_exact_package(tmp_path):
    package = make_package(tmp_path)
    approval = json.loads((package / 'approval.json').read_text())
    approval['weights_sha256'] = 'f' * 64
    (package / 'approval.json').write_text(json.dumps(approval))
    assert stages(inspect_package(package, nano_facts(), 'src'))['approved'] == 'FAIL'
    approval.update(weights_sha256=json.loads((package / 'manifest.json').read_text())['weights_sha256'],
                    approved_by='')
    (package / 'approval.json').write_text(json.dumps(approval))
    assert stages(inspect_package(package, nano_facts(), 'src'))['approved'] == 'FAIL'


def test_development_only_package_can_never_be_approved_or_activated(tmp_path):
    package = make_package(tmp_path, edit=lambda c: c.update(provenance=dict(development_only=True)))
    report = inspect_package(package, nano_facts(), 'src')
    assert report['development_only'] is True
    assert stages(report)['approved'] == 'FAIL' and not report['activatable']
    with pytest.raises(ModelIncompatible):
        activate(tmp_path, package.name, nano_facts(), 'src')


def test_activation_is_explicit_and_never_falls_back(tmp_path):
    make_package(tmp_path, 'first')
    make_package(tmp_path, 'second')
    assert active_package_missing(tmp_path)
    activate(tmp_path, 'first', nano_facts(), 'src')
    assert active_package(tmp_path, nano_facts(), 'src')['package_id'] == 'first'
    state = inventory(tmp_path, nano_facts(), 'src')
    assert state['state'] == 'ACTIVE' and state['active'] == 'first'
    by_id = dict((p['package_id'], p) for p in state['packages'])
    assert by_id['first']['stages']['active']['status'] == 'PASS'
    assert by_id['second']['stages']['active']['status'] == 'NOT_TESTED'
    # Removing or altering the active package fails closed instead of choosing 'second'.
    shutil.rmtree(str(tmp_path / 'first'))
    with pytest.raises(ModelNotDeployed):
        active_package(tmp_path, nano_facts(), 'src')


def active_package_missing(slot):
    try:
        active_package(slot, nano_facts(), 'src')
    except ModelNotDeployed:
        return True
    return False


def test_tampering_with_the_active_package_deactivates_it(tmp_path):
    package = make_package(tmp_path)
    activate(tmp_path, package.name, nano_facts(), 'src')
    (package / 'approval.json').unlink()
    with pytest.raises(ModelNotDeployed):
        active_package(tmp_path, nano_facts(), 'src')


def test_deactivate_removes_the_pointer_only(tmp_path):
    package = make_package(tmp_path)
    activate(tmp_path, package.name, nano_facts(), 'src')
    deactivate(tmp_path)
    assert package.is_dir() and not (tmp_path / 'active.json').exists()
    deactivate(tmp_path)  # idempotent


def test_install_copies_validated_packages_without_activating(tmp_path):
    incoming = make_package(tmp_path / 'usb', 'final-1')
    slot = tmp_path / 'slot'
    installed = install_package(incoming, slot)
    assert installed == slot / 'final-1' and (installed / 'weights.npz').is_file()
    assert not (slot / 'active.json').exists()
    with pytest.raises(ValueError):
        install_package(incoming, slot)  # never overwrite an installed package
    broken = make_package(tmp_path / 'usb', 'broken')
    (broken / 'weights.npz').write_bytes(b'not the declared weights')
    with pytest.raises(ValueError):
        install_package(broken, slot)
    assert not (slot / 'broken').exists()


def test_install_rejects_unsafe_package_names_and_extra_executables(tmp_path):
    incoming = make_package(tmp_path / 'usb', 'bad name')
    with pytest.raises(ValueError):
        install_package(incoming, tmp_path / 'slot')
    good = make_package(tmp_path / 'usb', 'okay')
    (good / 'run.sh').write_text('rm -rf /')
    with pytest.raises(ValueError):
        install_package(good, tmp_path / 'slot')


def test_inventory_lists_development_baseline_separately_and_never_active(tmp_path):
    bundle = tmp_path / 'gaitpt-bundle'
    bundle.mkdir()
    (bundle / 'manifest.json').write_text('{}')
    (bundle / 'weights.npz').write_bytes(b'x')
    state = inventory(tmp_path / 'slot', nano_facts(), 'src', baseline_bundle=bundle)
    baseline = state['development_baseline']
    assert baseline['development_only'] is True and baseline['stages']['discovered']['status'] == 'PASS'
    assert baseline['stages']['approved']['status'] == 'FAIL'
    assert baseline['stages']['active']['status'] == 'FAIL'
    assert state['state'] == 'MODEL_NOT_DEPLOYED' and state['active'] is None
    with pytest.raises(ModelIncompatible):
        activate(tmp_path / 'slot', 'development-baseline', nano_facts(), 'src')


def test_inventory_json_is_stable(tmp_path):
    make_package(tmp_path)
    state = inventory(tmp_path, nano_facts(), 'src')
    assert set(state) == {'schema', 'state', 'active', 'packages', 'development_baseline'}
    assert state['schema'] == 'nano-model-inventory-v1'
    assert set(state['packages'][0]) == {'package_id', 'family', 'variant', 'semantic_id',
                                         'development_only', 'stages', 'activatable', 'blockers'}
    json.dumps(state)
    assert model_slot.ACTIVE_FILE == 'active.json'


FIXTURE = Path(__file__).parent / 'fixtures/research_gaitpt_grew_parity_v1.json'


def replay(first_extra=0.0):
    cases = json.loads(FIXTURE.read_text())['cases']
    outputs = iter([[v + first_extra for v in cases[0]['walk_embedding']],
                    [v + 3e-3 for v in cases[1]['walk_embedding']]])
    return lambda crops: next(outputs)


def test_recorded_parity_is_evidence_the_stage_check_accepts(tmp_path):
    package = make_package(tmp_path, parity=False)
    record = record_parity(package, FIXTURE, replay(), nano_facts())
    assert record['environment']['hardware'].startswith('NVIDIA Jetson Nano')
    assert record['fixture_sha256'] and record['report']['passed'] is True
    assert stages(inspect_package(package, nano_facts(), 'src'))['parity_tested'] == 'PASS'
    assert (package / 'parity_report.json').stat().st_mode & 0o777 == 0o600


def test_workstation_parity_is_recorded_but_not_accepted_as_nano_evidence(tmp_path):
    package = make_package(tmp_path, parity=False)
    record_parity(package, FIXTURE, replay(), host_facts())
    assert stages(inspect_package(package, nano_facts(), 'src'))['parity_tested'] == 'WARNING'


def test_unrun_or_mismatched_parity_never_passes(tmp_path):
    package = make_package(tmp_path, parity=False)
    record_parity(package, FIXTURE, None, nano_facts())  # embeddings cannot be compared
    assert stages(inspect_package(package, nano_facts(), 'src'))['parity_tested'] == 'FAIL'
    record_parity(package, FIXTURE, replay(first_extra=0.5), nano_facts())  # numerically wrong
    assert stages(inspect_package(package, nano_facts(), 'src'))['parity_tested'] == 'FAIL'


def test_parity_refuses_a_fixture_from_a_different_model(tmp_path):
    package = make_package(tmp_path, parity=False,
                           edit=lambda c: c['model'].update(checkpoint_sha256='1' * 64))
    with pytest.raises(ValueError):
        record_parity(package, FIXTURE, replay(), nano_facts())
    assert not (package / 'parity_report.json').exists()

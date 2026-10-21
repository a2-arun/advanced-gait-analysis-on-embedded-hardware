#!/usr/bin/env python3
"""Assemble / verify the portable Jetson Nano lab kit (standard library only, Python 3.6+).

  python scripts/build_lab_kit.py build --output dist/jetson-lab-kit [--archive]
  python scripts/build_lab_kit.py verify dist/jetson-lab-kit        # also works on the Nano

The kit is an allowlist of source, templates and documents. It never contains a final model,
databases, recordings, datasets, secrets, wheels or checkpoints.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 'nano-lab-kit-v1'
REQUIRED_DOCS = (
    'docs/ARCHITECTURE.md', 'docs/SETUP.md', 'docs/JETSON_NANO_LAB_RUNBOOK.md',
    'docs/FINAL_MODEL_ACTIVATION_RUNBOOK.md', 'docs/JETSON_NANO_PRELAB_READINESS_REPORT.md',
    'docs/RESEARCH_TO_JETSON_MODEL_INTEGRATION.md', 'docs/RESEARCH_MODEL_DEPLOYMENT_CHECKLIST.md',
    'docs/gait_verification/GAIT_VERIFICATION_REARCHITECTURE_HANDOFF.md',
    'docs/gait_verification/GAIT_VERIFICATION_TECHNICAL_PRD.md')
TOP_FILES = ('README.md', 'AGENTS.md', 'config.yaml', 'requirements-jetson.txt', 'pytest.ini')
SCRIPTS = ('lab.py', 'gait.py', 'enroll.py', 'identify.py', 'verify.py', 'check_nano.py',
           'test_camera.py', 'build_lab_kit.py', 'make_parity_fixture.py')  # last: imported by tests
GENERATED = ('MANIFEST.sha256', 'MANIFEST.json', 'DEPENDENCIES.md', 'LAB_KIT_README.md')
FORBIDDEN_SUFFIXES = ('.db', '.sqlite', '.sqlite3', '.npz', '.npy', '.pth', '.pt', '.ckpt', '.onnx',
                      '.engine', '.tflite', '.safetensors', '.pkl', '.pickle', '.mp4', '.avi', '.mkv',
                      '.mov', '.whl', '.pem', '.key', '.p12', '.pfx')
FORBIDDEN_NAMES = ('.env', 'id_rsa', 'id_ed25519', 'credentials.json', 'recordings.json',
                   'active.json', 'approval.json', 'parity_report.json')
SECRET_PATTERNS = (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'), re.compile(r'AKIA[0-9A-Z]{16}'),
                   re.compile(r'gh[pousr]_[A-Za-z0-9]{36}'), re.compile(r'sk-[A-Za-z0-9]{32,}'))
SYSTEM_COMPONENTS = ('opencv', 'ffmpeg/ffprobe/v4l-utils')  # shipped by JetPack/apt, not wheels
EXTERNAL = (  # not obtainable from this repository; the operator must source and verify them
    ('torch', 'NVIDIA PyTorch 1.10.0 wheel for JetPack 4.6 / Python 3.6 / aarch64'),
    ('mediapipe', 'community MediaPipe 0.8.5 aarch64 wheel (experimental) with pose .tflite assets'),
    ('opencv', 'JetPack system OpenCV 4.1.1 (keep; do not pip install)'),
    ('ffmpeg/ffprobe/v4l-utils', 'apt packages from the Ubuntu 18.04 / JetPack repositories'))


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_files(root):
    root = Path(root)
    chosen = [n for n in TOP_FILES if (root / n).is_file()]
    chosen += sorted(str(p.relative_to(root)) for p in (root / 'src').rglob('*.py'))
    chosen += [('scripts/' + n) for n in SCRIPTS if (root / 'scripts' / n).is_file()]
    tests = root / 'tests' / 'verification'
    if tests.is_dir():
        chosen += sorted(str(p.relative_to(root)) for p in tests.glob('*.py'))
        chosen += sorted(str(p.relative_to(root)) for p in (tests / 'fixtures').glob('*.json'))
    chosen += [n for n in REQUIRED_DOCS if (root / n).is_file()]
    return chosen


def forbidden_files(directory):
    """Relative names that must never ship: data/model/media/key types or secret-looking content."""
    directory, bad = Path(directory), []
    for path in sorted(p for p in directory.rglob('*') if p.is_file()):
        name = str(path.relative_to(directory))
        flagged = path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES
        if not flagged and path.stat().st_size < 4 * 1024 * 1024:
            text = path.read_bytes().decode('utf8', 'ignore')
            flagged = any(pattern.search(text) for pattern in SECRET_PATTERNS)
        if flagged:
            bad.append(name)
    return bad


def dependency_report(root):
    root = Path(root)
    cache = root / '.cache'
    wheels = []
    listing = cache / 'nano-package-metadata-audit.json'
    for item in (json.loads(listing.read_text()) if listing.is_file() else []):
        local = cache / 'nano-dependency-wheels' / item['wheel']
        status = 'MISSING' if not local.is_file() else (
            'PRESENT_VERIFIED' if sha256(local) == item['sha256'] else 'HASH_MISMATCH')
        wheels.append(dict(name=item['name'], version=item['version'], wheel=item['wheel'],
                           sha256=item['sha256'], bytes=item['bytes'], offline_status=status))
    installers = sorted(p.name for p in (cache / 'installers').glob('*.whl')) \
        if (cache / 'installers').is_dir() else []
    return dict(wheels=wheels, local_installer_wheels=installers, external=[
        dict(name=n, what=w, offline_status=(
            'NOT_PACKAGED (check on the device)' if n in SYSTEM_COMPONENTS else
            'PRESENT_UNVERIFIED' if any(n in i.lower() for i in installers) else 'MISSING'))
        for n, w in EXTERNAL])


def dependencies_markdown(report):
    lines = ['# Dependencies for the original Jetson Nano (offline availability)', '',
             'This is a checklist, not an installer: the kit ships no wheels and no package manager',
             'commands are run for you. Obtain each item on a networked machine, verify its SHA-256, and',
             'follow `docs/SETUP.md` / `README.md` on the Nano. Wheels are aarch64 + Python 3.6 only;',
             'x86_64 wheels from a workstation are never substitutes.', '',
             '## Pure-Python / manylinux2014 aarch64 wheels from `requirements-jetson.txt`', '',
             '| wheel | bytes | sha256 | offline status |', '|---|---|---|---|']
    for item in report['wheels']:
        lines.append('| {wheel} | {bytes} | `{sha256}` | {offline_status} |'.format(**item))
    if not report['wheels']:
        lines.append('| (no wheel listing found in `.cache/nano-package-metadata-audit.json`) | | | UNKNOWN |')
    lines += ['', '## Platform components (not included in the kit)', '', '| component | note | offline status |',
              '|---|---|---|']
    for item in report['external']:
        lines.append('| {name} | {what} | {offline_status} |'.format(**item))
    lines += ['', 'Items marked MISSING were not present in the build machine\'s `.cache`; they are not',
              'included here and have not been downloaded by this kit.', '']
    return '\n'.join(lines)


README = """# Jetson Nano lab kit

Generated {created} from {source}. The finalized gait model does NOT exist yet and is not included;
no databases, recordings, datasets, secrets, wheels or checkpoints are packaged.

## 1. Verify the kit after transfer
    python3 scripts/build_lab_kit.py verify .          # stdlib only, works under Python 3.6
or `sha256sum -c MANIFEST.sha256` from the kit root.

## 2. Transfer
USB: copy `<kit>.tar.gz` and `<kit>.tar.gz.sha256`; on the Nano run
`sha256sum -c <kit>.tar.gz.sha256 && tar xzf <kit>.tar.gz`.
SSH: `scp <kit>.tar.gz <kit>.tar.gz.sha256 <user>@<nano-ip>:~/` then the same two commands.

## 3. Use
Follow `docs/JETSON_NANO_LAB_RUNBOOK.md` (start with `python3 scripts/lab.py preflight`).
Dependencies: `DEPENDENCIES.md`. Later model installation: `docs/FINAL_MODEL_ACTIVATION_RUNBOOK.md`.
"""


def git_info(root):
    def run(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root)] + list(args),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = run('status', '--porcelain')
    return dict(branch=run('branch', '--show-current'), commit=run('rev-parse', 'HEAD'),
                uncommitted_changes=None if status is None else bool(status))


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_kit(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    missing = [n for n in REQUIRED_DOCS if not (root / n).is_file()]
    if missing:
        raise ValueError('Required documents are missing: ' + ', '.join(missing))
    try:
        for name in selected_files(root):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(root / name), str(target))
        bad = forbidden_files(output)
        if bad:
            raise ValueError('Forbidden content would ship: ' + ', '.join(bad))
        created = datetime.now(timezone.utc).isoformat(timespec='seconds')
        source = git_info(root)
        write_text(output / 'DEPENDENCIES.md', dependencies_markdown(dependency_report(root)))
        write_text(output / 'LAB_KIT_README.md', README.format(created=created, source='{branch}@{commit}'.format(**source)))
        hashes = sorted((str(p.relative_to(output)), sha256(p)) for p in output.rglob('*') if p.is_file())
        manifest = dict(schema=SCHEMA, created_at=created, source=source, files=len(hashes) + 1,  # + MANIFEST.json
                        bytes=sum((output / n).stat().st_size for n, _ in hashes),
                        contains_final_model=False, contains_biometric_data=False,
                        excluded=['data/', '.cache/', 'outputs/', 'models/', 'legacy/', 'CONTEXT.md',
                                  'databases', 'recordings', 'checkpoints', 'wheels', 'secrets'])
        write_text(output / 'MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        hashes = sorted(hashes + [('MANIFEST.json', sha256(output / 'MANIFEST.json'))])
        write_text(output / 'MANIFEST.sha256', ''.join('{}  {}\n'.format(h, n) for n, h in hashes))
    except BaseException:
        shutil.rmtree(str(output), ignore_errors=True)
        raise
    return manifest


def verify_kit(directory):
    directory = Path(directory)
    recorded = {}
    for line in (directory / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        recorded[name] = digest
    present = sorted(str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()
                     and p.name != 'MANIFEST.sha256')
    missing = sorted(n for n in recorded if not (directory / n).is_file())
    modified = sorted(n for n in recorded if (directory / n).is_file() and sha256(directory / n) != recorded[n])
    unexpected = [n for n in present if n not in recorded]
    return dict(ok=not (missing or modified or unexpected), missing=missing, modified=modified,
                unexpected=unexpected)


def make_archive(directory):
    directory = Path(directory)
    archive = directory.parent / (directory.name + '.tar.gz')
    with tarfile.open(str(archive), 'w:gz') as tar:
        for path in sorted(p for p in directory.rglob('*') if p.is_file()):
            tar.add(str(path), arcname=directory.name + '/' + str(path.relative_to(directory)))
    write_text(Path(str(archive) + '.sha256'), '{}  {}\n'.format(sha256(archive), archive.name))
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command')
    build = commands.add_parser('build')
    build.add_argument('--output', default=None)
    build.add_argument('--archive', action='store_true')
    commands.add_parser('verify').add_argument('directory')
    args = parser.parse_args(argv)
    if args.command == 'build':
        output = Path(args.output or ROOT / 'dist' / ('jetson-lab-kit-' + datetime.now(timezone.utc).strftime('%Y%m%d')))
        manifest = build_kit(ROOT, output)
        print(json.dumps(dict(output=str(output), files=manifest['files'], bytes=manifest['bytes'],
                              archive=str(make_archive(output)) if args.archive else None), indent=2))
        return 0
    if args.command == 'verify':
        result = verify_kit(args.directory)
        print(json.dumps(result, indent=2))
        return 0 if result['ok'] else 2
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())

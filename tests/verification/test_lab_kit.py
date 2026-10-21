import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from scripts import build_lab_kit as kit

REAL_ROOT = Path(__file__).resolve().parents[2]


def mini_repo(root):
    """A throwaway repository containing allowlisted files plus things that must never ship."""
    files = {'src/__init__.py': '', 'src/verification/__init__.py': '', 'src/verification/lab.py': 'x = 1\n',
             'scripts/lab.py': 'print(1)\n', 'scripts/build_lab_kit.py': 'print(2)\n',
             'config.yaml': 'schema_version: 2\n', 'requirements-jetson.txt': 'numpy==1.19.4\n',
             'README.md': 'readme\n', 'tests/verification/test_x.py': 'def test_x():\n    pass\n'}
    for name in kit.REQUIRED_DOCS:
        files[name] = 'doc ' + name + '\n'
    forbidden = {'data/verification-v2.db': 'db', 'data/recordings.json': '{}', 'data/raw/camera/walk.mp4': 'v',
                 '.cache/gaitpt-grew-npz/weights.npz': 'w', 'outputs/lab-reports/r.json': '{}',
                 'models/checkpoint/full_hybrid_best.pth': 'p', 'legacy/old.py': 'old = 1\n',
                 'src/verification/secret.pem': 'pem material', 'src/stolen.db': 'x',
                 '.cache/model-slot/final/approval.json': '{}', 'CONTEXT.md': 'private notes'}
    for name, text in dict(files, **forbidden).items():
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def listing(out):
    return sorted(str(p.relative_to(out)) for p in Path(out).rglob('*') if p.is_file())


def test_build_includes_only_allowlisted_files_and_never_data_models_or_secrets(tmp_path):
    mini_repo(tmp_path / 'repo')
    out = tmp_path / 'kit'
    kit.build_kit(tmp_path / 'repo', out)
    names = listing(out)
    assert 'src/verification/lab.py' in names and 'scripts/lab.py' in names and 'config.yaml' in names
    for blocked in ('data/', '.cache', 'outputs/', 'models/', 'legacy/', '.pem', '.db', '.npz', '.mp4',
                    '.pth', 'CONTEXT.md', 'recordings.json', 'approval.json'):
        assert not any(blocked in n for n in names), blocked
    for generated in ('MANIFEST.sha256', 'MANIFEST.json', 'DEPENDENCIES.md', 'LAB_KIT_README.md'):
        assert generated in names


def test_manifest_hashes_every_file_and_verifies(tmp_path):
    mini_repo(tmp_path / 'repo')
    out = tmp_path / 'kit'
    manifest = kit.build_kit(tmp_path / 'repo', out)
    lines = (out / 'MANIFEST.sha256').read_text().splitlines()
    listed = dict(line.split('  ', 1)[::-1] for line in lines)
    for name in listing(out):
        if name != 'MANIFEST.sha256':
            assert listed[name] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert 'MANIFEST.sha256' not in listed
    assert manifest['contains_final_model'] is False and manifest['contains_biometric_data'] is False
    assert json.loads((out / 'MANIFEST.json').read_text())['files'] == len(listed)
    assert kit.verify_kit(out) == dict(ok=True, missing=[], modified=[], unexpected=[])


def test_verification_detects_tampering_missing_and_extra_files(tmp_path):
    mini_repo(tmp_path / 'repo')
    out = tmp_path / 'kit'
    kit.build_kit(tmp_path / 'repo', out)
    (out / 'src/verification/lab.py').write_text('x = 2\n')
    (out / 'README.md').unlink()
    (out / 'weights.npz').write_bytes(b'planted')
    result = kit.verify_kit(out)
    assert result['ok'] is False
    assert result['modified'] == ['src/verification/lab.py'] and result['missing'] == ['README.md']
    assert result['unexpected'] == ['weights.npz']


def test_build_refuses_existing_output_and_missing_required_docs(tmp_path):
    mini_repo(tmp_path / 'repo')
    (tmp_path / 'kit').mkdir()
    with pytest.raises(ValueError):
        kit.build_kit(tmp_path / 'repo', tmp_path / 'kit')
    (tmp_path / 'repo' / kit.REQUIRED_DOCS[0]).unlink()
    with pytest.raises(ValueError):
        kit.build_kit(tmp_path / 'repo', tmp_path / 'kit2')
    assert not (tmp_path / 'kit2').exists()


@pytest.mark.parametrize('name,content', [('weights.npz', b'x'), ('a/biometrics.db', b'x'),
                                          ('clip.mp4', b'x'), ('id.pem', b'x'),
                                          ('model.pth', b'x'), ('nano.whl', b'x'),
                                          ('notes.py', b'-----BEGIN RSA ' + b'PRIVATE KEY-----'),
                                          ('keys.py', b'AKIA' + b'ABCDEFGHIJKLMNOP')])
def test_guard_rejects_forbidden_content_anywhere_in_a_kit(tmp_path, name, content):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    assert kit.forbidden_files(tmp_path) == [name]


def test_dependency_report_lists_missing_offline_wheels_without_inventing_an_installer(tmp_path):
    mini_repo(tmp_path / 'repo')
    cache = tmp_path / 'repo' / '.cache'
    (cache / 'nano-dependency-wheels').mkdir(parents=True)
    (cache / 'nano-package-metadata-audit.json').write_text(json.dumps([dict(
        name='numpy', version='1.19.4', wheel='numpy-1.19.4-cp36-cp36m-manylinux2014_aarch64.whl',
        sha256='a' * 64, bytes=12185401, metadata_source='https://pypi.org/pypi/numpy/1.19.4/json')]))
    out = tmp_path / 'kit'
    kit.build_kit(tmp_path / 'repo', out)
    text = (out / 'DEPENDENCIES.md').read_text()
    assert 'numpy-1.19.4-cp36-cp36m-manylinux2014_aarch64.whl' in text and 'MISSING' in text
    assert 'torch' in text and 'mediapipe' in text and 'not included' in text.lower()
    assert 'not an installer' in text.lower()
    assert not any(n.endswith('.whl') for n in listing(out))
    (cache / 'nano-dependency-wheels' / 'numpy-1.19.4-cp36-cp36m-manylinux2014_aarch64.whl').write_bytes(b'x')
    report = kit.dependency_report(tmp_path / 'repo')
    assert report['wheels'][0]['offline_status'] == 'HASH_MISMATCH'
    (cache / 'nano-package-metadata-audit.json').write_text(json.dumps([dict(
        name='numpy', version='1.19.4', wheel='numpy-1.19.4-cp36-cp36m-manylinux2014_aarch64.whl',
        sha256=hashlib.sha256(b'x').hexdigest(), bytes=1, metadata_source='x')]))
    assert kit.dependency_report(tmp_path / 'repo')['wheels'][0]['offline_status'] == 'PRESENT_VERIFIED'


def test_archive_is_a_tar_gz_of_exactly_the_kit(tmp_path):
    mini_repo(tmp_path / 'repo')
    out = tmp_path / 'kit'
    kit.build_kit(tmp_path / 'repo', out)
    archive = kit.make_archive(out)
    with tarfile.open(str(archive)) as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile())
    assert names == sorted('kit/' + n for n in listing(out))
    assert Path(str(archive) + '.sha256').read_text().split()[0] == hashlib.sha256(
        archive.read_bytes()).hexdigest()


def test_real_repository_kit_is_clean(tmp_path):
    out = tmp_path / 'kit'
    kit.build_kit(REAL_ROOT, out)
    names = listing(out)
    for needed in ('scripts/lab.py', 'scripts/gait.py', 'src/verification/model_slot.py',
                   'docs/JETSON_NANO_LAB_RUNBOOK.md', 'docs/FINAL_MODEL_ACTIVATION_RUNBOOK.md',
                   'requirements-jetson.txt', 'config.yaml'):
        assert needed in names, needed
    assert kit.forbidden_files(out) == [] and kit.verify_kit(out)['ok'] is True
    assert not any(n.endswith(('.db', '.npz', '.pth', '.pt', '.mp4', '.whl')) for n in names)

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from lab_helpers import make_package, nano_facts

from src.verification import camera_diag, capture, lab, model_slot, readiness
from src.verification.backend import LocalGaitPT
from src.verification.cli import configuration

PROBES = dict(memory_mb=3956, disk_free_mb=20000, display=dict(connected=True, source='DISPLAY'),
              video_nodes=[dict(path='/dev/video0', accessible=True)],
              ffmpeg=dict(ffmpeg='4.2.7', ffprobe='4.2.7', libx264=True, v4l2=True), v4l2_ctl=True,
              jetpack_package='4.6.1-b110')


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(dict(schema_version=2, source_dir='source', bundle_dir='bundle',
        device='cuda', database='data/v2.db', recordings='data/recordings.json', policy=None,
        events_dir='events', log_file='logs/operator.log', capture_dir='data/raw/camera',
        camera=dict(device_id=0, fps_target=30, resolution=[160, 120]))))
    monkeypatch.setattr(readiness, 'collect_facts', lambda cfg, device='/dev/video0': dict(
        nano_facts(), probes=PROBES))
    monkeypatch.setattr(capture, 'capture_video', lambda *a, **k: pytest.fail('unexpected recording'))
    monkeypatch.setattr(LocalGaitPT, 'validate_artifacts', staticmethod(lambda s, b: {}))
    return path


def run(env, capsys, *argv):
    code = lab.main(['--config', str(env)] + list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_configuration_gains_lab_defaults_resolved_next_to_the_file(env):
    cfg = configuration(env)
    base = env.parent
    assert cfg['model_slot_dir'] == str(base / '.cache/model-slot')
    assert cfg['diagnostics_dir'] == str(base / 'data/diagnostics')
    assert cfg['reports_dir'] == str(base / 'outputs/lab-reports')


def test_no_command_prints_help_and_fails(env, capsys):
    code, out, err = run(env, capsys)
    assert code == 2 and 'preflight' in out


def test_preflight_saves_json_and_markdown_and_creates_nothing_else(env, capsys, tmp_path):
    code, out, err = run(env, capsys, 'preflight')
    assert code == 0 and 'MODEL_NOT_DEPLOYED' in out
    reports = list((tmp_path / 'outputs/lab-reports').glob('preflight-*'))
    assert sorted(p.suffix for p in reports) == ['.json', '.md']
    assert json.loads([p for p in reports if p.suffix == '.json'][0].read_text())['schema'] == 'nano-preflight-v1'
    assert not (tmp_path / 'data').exists()  # no database, registry, recording or diagnostics


def test_preflight_json_mode_and_failure_exit_code(env, capsys, monkeypatch):
    code, out, err = run(env, capsys, 'preflight', '--json')
    assert json.loads(out)['summary']['hardware_overall'] == 'PASS'
    monkeypatch.setattr(readiness, 'collect_facts', lambda cfg, device='/dev/video0': dict(
        nano_facts(cuda_available=False), probes=PROBES))
    code, out, err = run(env, capsys, 'preflight', '--json')
    assert code == 2 and json.loads(out)['summary']['overall'] == 'FAIL'


@pytest.mark.parametrize('argv', [['camera', 'record', '--seconds', '0'],
                                  ['camera', 'record', '--seconds', '31'],
                                  ['camera', 'record', '--device', '/etc/passwd'],
                                  ['camera', 'record', '--width', '0']])
def test_camera_record_validates_arguments_before_touching_the_camera(env, capsys, argv):
    code, out, err = run(env, capsys, *argv)
    assert code == 2 and 'ERROR' in err and out == ''


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg required')
def test_camera_record_is_explicit_and_stays_in_the_diagnostics_folder(env, capsys, monkeypatch, tmp_path):
    def synthetic_camera(path, camera, seconds, device=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=30',
                        '-t', str(seconds), '-c:v', 'mpeg4', '-threads', '1', str(path)], check=True)
    monkeypatch.setattr(capture, 'capture_video', synthetic_camera)
    code, out, err = run(env, capsys, 'camera', 'record', '--seconds', '2')
    report = json.loads(out)
    assert code == 0 and report['analysis']['status'] == 'PASS'
    assert Path(report['video']).parent == tmp_path / 'data/diagnostics/camera'
    assert not (tmp_path / 'data/raw').exists() and not (tmp_path / 'data/recordings.json').exists()


def test_camera_list_and_analyze(env, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(camera_diag, 'list_cameras', lambda nodes=None: dict(nodes=[], modes_status='NOT_TESTED'))
    code, out, err = run(env, capsys, 'camera', 'list')
    assert code == 0 and json.loads(out)['modes_status'] == 'NOT_TESTED'
    code, out, err = run(env, capsys, 'camera', 'analyze', '--video', str(tmp_path / 'missing.mp4'))
    assert code == 2 and json.loads(out)['status'] == 'FAIL'


def test_db_init_check_and_no_fake_enrollment(env, capsys, tmp_path):
    code, out, err = run(env, capsys, 'db', 'check')
    assert json.loads(out)['state'] == 'ABSENT' and not (tmp_path / 'data').exists()
    code, out, err = run(env, capsys, 'db', 'init')
    status = json.loads(out)
    assert code == 0 and status['state'] == 'READY' and status['references'] == 0
    code, out, err = run(env, capsys, 'db', 'check')
    assert json.loads(out)['references'] == 0  # init never inserts synthetic participants


def test_model_status_reports_not_deployed_and_baseline_separately(env, capsys, tmp_path):
    (tmp_path / 'bundle').mkdir()
    (tmp_path / 'bundle/manifest.json').write_text('{}')
    (tmp_path / 'bundle/weights.npz').write_bytes(b'x')
    code, out, err = run(env, capsys, 'model', 'status')
    state = json.loads(out)
    assert code == 0 and state['state'] == 'MODEL_NOT_DEPLOYED' and state['active'] is None
    assert state['development_baseline']['development_only'] is True


def test_model_install_inspect_activate_and_deactivate_flow(env, capsys, tmp_path):
    incoming = make_package(tmp_path / 'usb', 'final-1')
    code, out, err = run(env, capsys, 'model', 'install', str(incoming))
    assert code == 0 and not (tmp_path / '.cache/model-slot/active.json').exists()
    code, out, err = run(env, capsys, 'model', 'inspect', 'final-1')
    report = json.loads(out)
    assert report['stages']['structurally_valid']['status'] == 'PASS'
    assert report['stages']['runtime_compatible']['status'] == 'PASS'  # facts are the pretend Nano
    code, out, err = run(env, capsys, 'model', 'activate', 'final-1')
    assert code == 0 and json.loads(out)['package_id'] == 'final-1'
    code, out, err = run(env, capsys, 'model', 'status')
    assert json.loads(out)['active'] == 'final-1'
    code, out, err = run(env, capsys, 'model', 'deactivate')
    assert code == 0 and json.loads(run(env, capsys, 'model', 'status')[1])['active'] is None


def test_activation_without_qualification_is_refused_with_reasons(env, capsys, tmp_path):
    make_package(tmp_path / '.cache/model-slot', 'unapproved', approve=False)
    code, out, err = run(env, capsys, 'model', 'activate', 'unapproved')
    assert code == 2 and 'approved' in err
    code, out, err = run(env, capsys, 'model', 'activate', 'development-baseline')
    assert code == 2


def test_model_parity_records_evidence_using_the_packages_own_backend(env, capsys, tmp_path, monkeypatch):
    make_package(tmp_path / '.cache/model-slot', 'final-1', parity=False)
    fixture = Path(__file__).parent / 'fixtures/research_gaitpt_grew_parity_v1.json'
    cases = json.loads(fixture.read_text())['cases']
    outputs = iter([cases[0]['walk_embedding'], [v + 3e-3 for v in cases[1]['walk_embedding']]])
    monkeypatch.setattr(lab, 'package_encoder', lambda cfg, package: (lambda crops: np.asarray(next(outputs))))
    code, out, err = run(env, capsys, 'model', 'parity', 'final-1', '--fixture', str(fixture))
    assert code == 0 and json.loads(out)['report']['passed'] is True
    assert model_slot.inspect_package(tmp_path / '.cache/model-slot/final-1', nano_facts(), 'x')[
        'stages']['parity_tested']['status'] == 'PASS'


def test_pose_check_refuses_unknown_backend_and_extract_needs_an_explicit_file(env, capsys, tmp_path):
    code, out, err = run(env, capsys, 'pose', 'check', '--backend', 'rtmpose-m')
    assert code == 2 and 'mediapipe-solutions-single-person-lite' in err
    code, out, err = run(env, capsys, 'pose', 'extract', '--video', str(tmp_path / 'none.mp4'))
    assert code == 2 and 'ERROR' in err


def test_display_status_renders_not_ready_text_and_result_screen(env, capsys, tmp_path):
    code, out, err = run(env, capsys, 'display', 'status', '--render', 'text', '--no-color')
    assert code == 0 and 'NOT READY: MODEL NOT DEPLOYED' in out and 'TRUE' not in out
    event = tmp_path / 'event.json'
    event.write_text(json.dumps(dict(schema_version='gait-verification-v2', request_id='r',
                                     claimed_subject_id='S001', status='MODEL_NOT_DEPLOYED',
                                     verdict=None, message='m')))
    code, out, err = run(env, capsys, 'display', 'result', '--event', str(event), '--render', 'text',
                         '--no-color')
    assert code == 0 and 'MODEL NOT DEPLOYED' in out and 'S001' in out


def test_display_window_without_a_desktop_is_an_error_not_a_crash(env, capsys, monkeypatch):
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setattr('sys.platform', 'linux')
    code, out, err = run(env, capsys, 'display', 'status', '--render', 'window')
    assert code == 2 and 'DISPLAY' in err


def test_display_watch_redraws_a_bounded_number_of_times(env, capsys):
    code, out, err = run(env, capsys, 'display', 'status', '--render', 'text', '--no-color',
                         '--watch', '0.01', '--iterations', '3')
    assert code == 0 and out.count('Gait verification') == 3


def test_source_avoids_apis_that_python_3_6_does_not_have():
    # The grammar check cannot see library calls; these are the 3.7+ additions that bite.
    import ast
    root = Path(lab.__file__).parents[2]
    banned_keywords = {'add_subparsers': {'required'}, 'run': {'capture_output', 'text'}}
    banned_names = {'fromisoformat', 'is_relative_to', 'nullcontext', 'time_ns', 'removeprefix'}
    for path in list((root / 'src/verification').glob('*.py')) + list((root / 'scripts').glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                used = {k.arg for k in node.keywords}
                assert not used & banned_keywords.get(name, set()), (path.name, name, used)
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_names, (path.name, node.attr)

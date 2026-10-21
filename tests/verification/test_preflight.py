import json
import stat

import pytest
from lab_helpers import host_facts, make_package, nano_facts

from src.verification import readiness
from src.verification.readiness import STATUSES, run_preflight, write_reports

GOOD_PROBES = dict(memory_mb=3956, disk_free_mb=20000, display=dict(connected=True, source='DISPLAY'),
                   video_nodes=[dict(path='/dev/video0', accessible=True)],
                   ffmpeg=dict(ffmpeg='4.2.7', ffprobe='4.2.7', libx264=True, v4l2=True),
                   v4l2_ctl=True, jetpack_package='4.6.1-b110', nvpmodel='MAXN')


def facts(base=None, **changes):
    value = dict(base or nano_facts())
    value['probes'] = dict(GOOD_PROBES, **changes)
    return value


def cfg(tmp_path, slot='slot', **extra):
    return dict(database=str(tmp_path / 'data' / 'v2.db'), model_slot_dir=str(tmp_path / slot),
                source_dir=str(tmp_path / 'source'), bundle_dir=str(tmp_path / 'bundle'),
                diagnostics_dir=str(tmp_path / 'diag'), **extra)


def by_id(report):
    return dict((c['id'], c) for c in report['checks'])


def test_every_result_uses_the_five_status_words_and_names_its_scope(tmp_path):
    report = run_preflight(cfg(tmp_path), facts())
    assert STATUSES == ('PASS', 'FAIL', 'WARNING', 'NOT_TESTED', 'BLOCKED')
    for check in report['checks']:
        assert check['status'] in STATUSES
        assert check['scope'] in ('host-independent', 'requires-nano')
        assert check['detail'] and check['title']
    ids = set(by_id(report))
    assert {'board', 'os_l4t_jetpack', 'python', 'cuda', 'pytorch', 'memory', 'disk', 'ffmpeg',
            'ffprobe', 'opencv', 'mediapipe', 'usb_camera', 'display', 'database_dir',
            'model_artifacts', 'package_compatibility', 'source_python36_grammar'} <= ids
    assert len(ids) == len(report['checks'])  # unique ids


def test_documented_nano_profile_passes_hardware_checks_but_model_stays_blocked(tmp_path):
    report = run_preflight(cfg(tmp_path), facts())
    checks = by_id(report)
    for name in ('board', 'os_l4t_jetpack', 'python', 'cuda', 'pytorch', 'memory', 'ffmpeg',
                 'ffprobe', 'opencv', 'mediapipe', 'usb_camera', 'display', 'package_compatibility'):
        assert checks[name]['status'] == 'PASS', (name, checks[name])
    assert checks['model_artifacts']['status'] == 'BLOCKED'
    assert 'MODEL_NOT_DEPLOYED' in checks['model_artifacts']['detail']
    assert report['summary']['hardware_overall'] == 'PASS'
    assert report['summary']['model_overall'] == 'BLOCKED'
    assert report['summary']['overall'] != 'PASS'  # cannot be fully ready without a final model


def test_workstation_never_reports_nano_checks_as_pass(tmp_path):
    report = run_preflight(cfg(tmp_path), facts(host_facts()))
    for check in report['checks']:
        if check['scope'] == 'requires-nano':
            assert check['status'] == 'NOT_TESTED', check
    assert by_id(report)['disk']['status'] == 'PASS'
    assert report['host']['on_nano'] is False
    assert report['summary']['hardware_overall'] == 'INCOMPLETE'


def missing(name):
    def edit(f):
        f['packages'].pop(name, None)
        f['import_errors'][name] = 'import failed'
    return edit


@pytest.mark.parametrize('check,edit', [
    ('board', lambda f: f.update(hardware='NVIDIA Jetson AGX Orin')),
    ('python', lambda f: f.update(python='3.8.10')),
    ('cuda', lambda f: f.update(cuda_available=False)),
    ('cuda', lambda f: f.update(cuda='11.4')),
    ('os_l4t_jetpack', lambda f: f.update(ubuntu='20.04')),
    ('os_l4t_jetpack', lambda f: f.update(l4t='# R35 (release), REVISION: 3.1')),
    ('opencv', missing('opencv')),
    ('pytorch', missing('torch')),
    ('mediapipe', missing('mediapipe')),
])
def test_nano_requirement_violations_fail(tmp_path, check, edit):
    base = nano_facts()
    edit(base)
    assert by_id(run_preflight(cfg(tmp_path), facts(base)))[check]['status'] == 'FAIL'


def test_probe_failures(tmp_path):
    cases = [('memory', dict(memory_mb=1500), 'FAIL'),
             ('disk', dict(disk_free_mb=300), 'FAIL'),
             ('disk', dict(disk_free_mb=3000), 'WARNING'),
             ('ffmpeg', dict(ffmpeg=dict(ffmpeg=None, ffprobe=None, libx264=False, v4l2=False)), 'FAIL'),
             ('ffprobe', dict(ffmpeg=dict(ffmpeg='4', ffprobe=None, libx264=True, v4l2=True)), 'FAIL'),
             ('usb_camera', dict(video_nodes=[]), 'FAIL'),
             ('usb_camera', dict(video_nodes=[dict(path='/dev/video0', accessible=False)]), 'FAIL'),
             ('display', dict(display=dict(connected=False, source=None)), 'WARNING')]
    for name, probe, expected in cases:
        assert by_id(run_preflight(cfg(tmp_path), facts(**probe)))[name]['status'] == expected, name


def test_package_pin_mismatch_fails_with_names(tmp_path):
    base = nano_facts()
    base['packages'] = dict(base['packages'], numpy='1.24.0')
    check = by_id(run_preflight(cfg(tmp_path), facts(base)))['package_compatibility']
    assert check['status'] == 'FAIL' and 'numpy' in check['detail']


def test_database_directory_states(tmp_path):
    ok = by_id(run_preflight(cfg(tmp_path), facts()))['database_dir']
    assert ok['status'] == 'PASS' and 'not initialized' in ok['detail']
    assert not (tmp_path / 'data').exists()  # preflight never creates the database
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'v2.db').write_bytes(b'garbage' * 50)
    assert by_id(run_preflight(cfg(tmp_path), facts()))['database_dir']['status'] == 'FAIL'


def test_unwritable_database_directory_fails(tmp_path):
    locked = tmp_path / 'locked'
    locked.mkdir()
    locked.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        value = cfg(tmp_path, )
        value['database'] = str(locked / 'sub' / 'v2.db')
        import os
        if os.access(str(locked), os.W_OK):
            pytest.skip('running with privileges that ignore directory permissions')
        assert by_id(run_preflight(value, facts()))['database_dir']['status'] == 'FAIL'
    finally:
        locked.chmod(stat.S_IRWXU)


def test_model_artifact_check_reports_the_stage_without_collapsing_it(tmp_path, monkeypatch):
    from src.verification.backend import LocalGaitPT
    monkeypatch.setattr(LocalGaitPT, 'validate_artifacts', staticmethod(lambda s, b: {}))
    make_package(tmp_path / 'slot', 'final-1', parity=False)
    check = by_id(run_preflight(cfg(tmp_path), facts()))['model_artifacts']
    assert check['status'] == 'BLOCKED'
    assert 'final-1' in check['detail'] and 'parity_tested' in check['detail']


def test_development_baseline_presence_is_labeled_not_deployed(tmp_path):
    (tmp_path / 'bundle').mkdir()
    (tmp_path / 'bundle' / 'manifest.json').write_text('{}')
    (tmp_path / 'bundle' / 'weights.npz').write_bytes(b'x')
    check = by_id(run_preflight(cfg(tmp_path), facts()))['model_artifacts']
    assert check['status'] == 'BLOCKED'
    assert 'development baseline' in check['detail'].lower() and 'not' in check['detail'].lower()


def test_active_model_is_the_only_pass_for_model_artifacts(tmp_path, monkeypatch):
    from src.verification.backend import LocalGaitPT
    from src.verification.model_slot import activate
    monkeypatch.setattr(LocalGaitPT, 'validate_artifacts', staticmethod(lambda s, b: {}))
    make_package(tmp_path / 'slot', 'final-1')
    activate(tmp_path / 'slot', 'final-1', nano_facts(), 'x')
    report = run_preflight(cfg(tmp_path), facts())
    assert by_id(report)['model_artifacts']['status'] == 'PASS'
    assert report['summary']['model_overall'] == 'PASS'


def test_preflight_does_not_open_a_camera(tmp_path, monkeypatch):
    from src.verification import capture
    monkeypatch.setattr(capture, 'capture_video',
                        lambda *a, **k: pytest.fail('preflight must not record'))
    run_preflight(cfg(tmp_path), facts())


def test_tool_versions_are_reduced_to_the_version_number():
    banner = 'ffmpeg version 4.2.7-0ubuntu0.1 Copyright (c) 2000-2022 the FFmpeg developers\nbuilt'
    assert readiness.tool_version(banner) == '4.2.7-0ubuntu0.1'
    assert readiness.tool_version('') is None
    assert readiness.tool_version('unexpected output') == 'unexpected output'


def test_python36_grammar_check_flags_unparseable_source(tmp_path):
    good = tmp_path / 'ok.py'
    good.write_text('x = 1\n')
    assert readiness.python36_grammar_errors([good]) == []
    bad = tmp_path / 'bad.py'
    bad.write_text('def f(:\n')
    assert readiness.python36_grammar_errors([bad]) == [str(bad)]


def test_reports_are_written_as_private_json_and_markdown(tmp_path):
    report = run_preflight(cfg(tmp_path), facts())
    json_path, md_path = write_reports(report, tmp_path / 'reports')
    assert json.loads(json_path.read_text())['schema'] == 'nano-preflight-v1'
    text = md_path.read_text()
    assert '| board |' in text
    assert 'does not prove' in text  # scope disclaimer is part of the readable report
    assert json_path.stat().st_mode & 0o777 == 0o600 and md_path.stat().st_mode & 0o777 == 0o600
    json_two, _ = write_reports(report, tmp_path / 'reports')
    assert json_two != json_path and json_path.exists()  # never silently overwrites evidence


def test_collect_facts_on_this_machine_is_complete_and_harmless(tmp_path):
    collected = readiness.collect_facts(cfg(tmp_path))
    assert {'memory_mb', 'disk_free_mb', 'display', 'video_nodes', 'ffmpeg'} <= set(collected['probes'])
    report = run_preflight(cfg(tmp_path), collected)
    assert all(c['status'] in STATUSES for c in report['checks'])

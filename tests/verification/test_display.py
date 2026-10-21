import numpy as np
import pytest
from lab_helpers import make_package, nano_facts

from src.verification import display
from src.verification.backend import LocalGaitPT
from src.verification.display import (render_image, render_text, result_screen, status_screen,
                                      gather_state)
from src.verification.service import MESSAGES, response
from src.verification.store import EnrollmentStore

PROBES = dict(video_nodes=[dict(path='/dev/video0', accessible=True)], display=dict(connected=True))


def cfg(tmp_path):
    return dict(database=str(tmp_path / 'data/v2.db'), model_slot_dir=str(tmp_path / 'slot'),
                source_dir=str(tmp_path / 'source'), bundle_dir=str(tmp_path / 'bundle'),
                diagnostics_dir=str(tmp_path / 'diag'), reports_dir=str(tmp_path / 'reports'),
                events_dir=str(tmp_path / 'events'), policy=None)


def state(tmp_path, **changes):
    return gather_state(cfg(tmp_path), lambda: dict(nano_facts(), probes=PROBES), nodes=changes.pop(
        'nodes', ['/dev/video0']))


def rows(screen):
    return dict((label, (value, level)) for label, value, level in screen['rows'])


def test_before_a_final_model_the_screen_says_not_ready_not_a_result(tmp_path):
    screen = status_screen(state(tmp_path))
    assert screen['headline'].startswith('NOT READY') and 'MODEL NOT DEPLOYED' in screen['headline']
    assert rows(screen)['Model'][0].startswith('NOT DEPLOYED')
    assert rows(screen)['Verification'][0].startswith('NOT READY')
    text = render_text(screen, color=False)
    assert 'TRUE' not in text and 'FALSE' not in text


def test_all_required_rows_are_present_and_database_enrollment_are_reported(tmp_path):
    EnrollmentStore(tmp_path / 'data/v2.db').add('S001', 't1', 'a' * 64, 'r', np.array([1., 0.]),
                                                  'operational')
    screen = status_screen(state(tmp_path))
    assert list(rows(screen)) == ['Hardware', 'Camera', 'Capture', 'Model', 'Database',
                                  'Enrollment', 'Verification']
    assert 'READY' in rows(screen)['Database'][0] and rows(screen)['Database'][1] == 'ok'
    assert '1 reference' in rows(screen)['Enrollment'][0]


def test_dev_baseline_is_labeled_and_does_not_make_the_model_ready(tmp_path):
    (tmp_path / 'bundle').mkdir()
    (tmp_path / 'bundle' / 'manifest.json').write_text('{}')
    (tmp_path / 'bundle' / 'weights.npz').write_bytes(b'x')
    screen = status_screen(state(tmp_path))
    value, level = rows(screen)['Model']
    assert 'dev baseline' in value.lower() and level == 'warn'
    assert screen['headline'].startswith('NOT READY')


def test_active_model_is_qualified_before_display(tmp_path, monkeypatch):
    from src.verification.model_slot import activate
    monkeypatch.setattr(LocalGaitPT, 'validate_artifacts', staticmethod(lambda s, b: {}))
    make_package(tmp_path / 'slot', 'final-1')
    activate(tmp_path / 'slot', 'final-1', nano_facts(), 'x')
    screen = status_screen(state(tmp_path))
    assert rows(screen)['Model'] == ('ACTIVE final-1', 'ok')
    (tmp_path / 'slot' / 'final-1' / 'approval.json').unlink()  # no longer qualifies
    assert rows(status_screen(state(tmp_path)))['Model'][0].startswith('NOT DEPLOYED')


def test_missing_camera_and_unready_database_are_visible(tmp_path):
    screen = status_screen(state(tmp_path, nodes=[]))
    assert rows(screen)['Camera'][1] == 'fail'
    assert rows(screen)['Database'][0].startswith('NOT INITIALIZED')


def test_model_readiness_does_not_load_facts_when_nothing_is_activated(tmp_path):
    def forbidden():
        raise AssertionError('torch/mediapipe probing is unnecessary without an activated model')
    gather_state(cfg(tmp_path), forbidden, nodes=[])


@pytest.mark.parametrize('result,headline,level', [
    (response('S001', 'OK', True), 'TRUE', 'ok'),
    (response('S001', 'OK', False), 'FALSE', 'fail'),
    (response('S001', 'POLICY_NOT_READY'), 'NO DECISION: POLICY NOT READY', 'warn'),
    (response('S001', 'INSUFFICIENT_GAIT'), 'INSUFFICIENT GAIT', 'warn'),
    (response('S001', 'CLAIM_NOT_ENROLLED'), 'NOT ENROLLED', 'warn'),
    (response('S001', 'MODEL_NOT_DEPLOYED'), 'MODEL NOT DEPLOYED', 'warn'),
    (response('S001', 'MODEL_INCOMPATIBLE'), 'MODEL INCOMPATIBLE', 'fail'),
    (response('S001', 'PROCESSING_ERROR'), 'PROCESSING FAILURE', 'fail'),
    (response('S001', 'INVALID_INPUT'), 'INVALID INPUT', 'fail')])
def test_every_documented_result_has_a_clear_screen(result, headline, level):
    screen = result_screen(result)
    assert screen['headline'] == headline and screen['level'] == level
    assert 'S001' in render_text(screen, color=False)  # the claimed identity is always shown


def test_every_service_status_has_a_screen_so_the_lcd_never_crashes():
    for status in MESSAGES:
        verdict = True if status == 'OK' else None
        assert result_screen(response('S001', status, verdict))['headline']


@pytest.mark.parametrize('bad', [
    dict(status='OK', verdict=None), dict(status='OK', verdict='yes'), dict(status='OK', verdict=1),
    dict(status='POLICY_NOT_READY', verdict=True), dict(status='MODEL_NOT_DEPLOYED', verdict=False),
    dict(status='NEW_STATUS', verdict=None), {}, None, 'TRUE'])
def test_a_boolean_is_shown_only_for_status_ok_with_a_real_boolean(bad):
    screen = result_screen(dict(claimed_subject_id='S001', **bad) if isinstance(bad, dict) else bad)
    assert screen['headline'] not in ('TRUE', 'FALSE') and screen['level'] == 'fail'
    assert screen['headline'] == 'UNRECOGNIZED RESULT'


def test_status_values_are_a_stable_closed_set():
    assert set(MESSAGES) == {'OK', 'CLAIM_NOT_ENROLLED', 'INSUFFICIENT_GAIT', 'INVALID_INPUT',
                             'PROCESSING_ERROR', 'POLICY_NOT_READY', 'MODEL_NOT_DEPLOYED',
                             'MODEL_INCOMPATIBLE'}
    assert set(response('S001', 'OK', True)) == {'schema_version', 'request_id', 'claimed_subject_id',
                                                 'status', 'verdict', 'message'}


def test_text_rendering_is_plain_without_color_and_never_overflows(tmp_path):
    screen = result_screen(response('S001', 'MODEL_NOT_DEPLOYED'))
    plain = render_text(screen, color=False, width=40)
    assert '\x1b' not in plain and all(len(line) <= 40 for line in plain.splitlines())
    assert '\x1b[' in render_text(screen, color=True, width=40)


def test_image_rendering_has_requested_size_and_nonblank_pixels():
    pytest.importorskip('cv2')
    image = render_image(result_screen(response('S001', 'MODEL_NOT_DEPLOYED')), 800, 480)
    assert image.shape == (480, 800, 3) and image.dtype == np.uint8 and image.std() > 5


def test_window_mode_refuses_without_a_display(tmp_path, monkeypatch):
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setattr(display.sys, 'platform', 'linux')
    with pytest.raises(RuntimeError):
        display.show_window(result_screen(response('S001', 'MODEL_NOT_DEPLOYED')))


def test_png_output_is_written_privately(tmp_path):
    pytest.importorskip('cv2')
    target = tmp_path / 'out' / 'screen.png'
    display.save_png(status_screen(state(tmp_path)), target, 800, 480)
    assert target.is_file() and target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        display.save_png(status_screen(state(tmp_path)), target, 800, 480)  # never overwrite

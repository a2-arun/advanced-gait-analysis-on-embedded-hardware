"""Small-HDMI-LCD status and result screens. Presentation only: it reads state and verification
results, never computes a decision, and shows TRUE/FALSE solely for status OK with a real boolean."""
import glob
import json
import os
import sys
from pathlib import Path

from .contract import identifier
from .dbcheck import database_status
from .environment import read_optional
from .model_slot import ACTIVE_FILE, inventory

LEVELS = dict(ok=(0, 200, 0), warn=(0, 165, 255), fail=(0, 0, 255), info=(255, 255, 255))  # BGR
ANSI = dict(ok='32', warn='33', fail='31', info='37')
RESULTS = {  # status -> (headline, level, hint)
    'POLICY_NOT_READY': ('NO DECISION: POLICY NOT READY', 'warn', 'No approved decision policy for this model.'),
    'INSUFFICIENT_GAIT': ('INSUFFICIENT GAIT', 'warn', 'Recapture a full-body walk.'),
    'CLAIM_NOT_ENROLLED': ('NOT ENROLLED', 'warn', 'Enroll this identity with the active model.'),
    'MODEL_NOT_DEPLOYED': ('MODEL NOT DEPLOYED', 'warn', 'No approved gait model is activated.'),
    'MODEL_INCOMPATIBLE': ('MODEL INCOMPATIBLE', 'fail', 'Model or preprocessing does not match.'),
    'PROCESSING_ERROR': ('PROCESSING FAILURE', 'fail', 'See the operator log.'),
    'INVALID_INPUT': ('INVALID INPUT', 'fail', 'Check the claimed ID and recording.'),
}


def latest_json(pattern):
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
    return None


def model_state(cfg, facts_provider):
    slot = Path(cfg['model_slot_dir'])
    bundle = Path(cfg.get('bundle_dir', ''))
    baseline = (bundle / 'manifest.json').is_file() and (bundle / 'weights.npz').is_file()
    if (slot / ACTIVE_FILE).is_file():  # only then are torch/mediapipe facts worth collecting
        found = inventory(slot, facts_provider(), cfg['source_dir'])
        if found['state'] == 'ACTIVE':
            return dict(kind='ACTIVE', id=found['active'], baseline=baseline)
        return dict(kind='UNQUALIFIED', id=None, baseline=baseline)
    return dict(kind='NOT_DEPLOYED', id=None, baseline=baseline)


def gather_state(cfg, facts_provider, nodes=None):
    nodes = sorted(glob.glob('/dev/video*')) if nodes is None else list(nodes)
    return dict(
        board=read_optional('/proc/device-tree/model') or None,
        preflight=latest_json(str(Path(cfg.get('reports_dir', '.')) / 'preflight-*.json')),
        camera_nodes=nodes,
        last_capture=latest_json(str(Path(cfg['diagnostics_dir']) / 'camera' / '*.diag.json')),
        model=model_state(cfg, facts_provider), database=database_status(cfg['database']),
        policy=cfg.get('policy'))


def status_screen(state):
    rows, model, db = [], state['model'], state['database']
    board, preflight = state['board'], state['preflight']
    if preflight:
        overall = preflight['summary']['hardware_overall']
        rows.append(('Hardware', '{} preflight {} {}'.format(board or 'not a Jetson', overall,
                                                               preflight['generated_at'][:16]),
                     dict(PASS='ok', FAIL='fail').get(overall, 'warn')))
    else:
        rows.append(('Hardware', '{}; preflight not run'.format(board or 'not a Jetson'), 'warn'))
    nodes = state['camera_nodes']
    rows.append(('Camera', '{} (not opened)'.format(', '.join(nodes)) if nodes else 'NO CAMERA NODE',
                 'ok' if nodes else 'fail'))
    capture = state['last_capture']
    status = capture['analysis']['status'] if capture else None
    rows.append(('Capture', '{} {}'.format(status, capture['created_at'][:16]) if capture
                 else 'no diagnostic recorded', dict(PASS='ok', FAIL='fail', WARNING='warn').get(status, 'info')))
    if model['kind'] == 'ACTIVE':
        rows.append(('Model', 'ACTIVE ' + model['id'], 'ok'))
    else:
        note = {'UNQUALIFIED': ' (active model failed qualification)'}.get(model['kind'], '')
        note += ' (dev baseline present, not final)' if model['baseline'] else ''
        rows.append(('Model', 'NOT DEPLOYED' + note, 'fail' if model['kind'] == 'UNQUALIFIED' else 'warn'))
    ready = db['state'] == 'READY'
    rows.append(('Database', 'READY schema v{}'.format(db['schema_version']) if ready else
                 'NOT INITIALIZED' if db['state'] == 'ABSENT' else db['state'],
                 'ok' if ready else 'warn' if db['state'] in ('ABSENT', 'NEEDS_MIGRATION', 'UNINITIALIZED')
                 else 'fail'))
    count = db['references'] if ready else 0
    rows.append(('Enrollment', '{} reference(s), {} subject(s)'.format(count, db['subjects'])
                 if ready else 'n/a', 'info' if count else 'warn'))
    if model['kind'] != 'ACTIVE':
        reason, level = 'MODEL NOT DEPLOYED', 'warn'
    elif not ready:
        reason, level = 'DATABASE', 'warn'
    elif not nodes:
        reason, level = 'NO CAMERA', 'fail'
    elif not count:
        reason, level = 'NO ENROLLMENT', 'warn'
    else:
        reason, level = None, 'ok'
    if reason:
        rows.append(('Verification', 'NOT READY: ' + reason, level))
        headline = 'NOT READY: ' + reason
    elif not state['policy']:
        rows.append(('Verification', 'NO DECISION POLICY (POLICY_NOT_READY)', 'warn'))
        headline, level = 'READY: NO DECISION POLICY', 'warn'
    else:
        rows.append(('Verification', 'READY (policy file configured; checked per request)', 'ok'))
        headline = 'READY'
    return dict(title='Gait verification', headline=headline, level=level, rows=rows,
                footer='Engineering status; not a biometric result.')


def result_screen(result):
    claim = '?'
    try:
        claim = identifier(result.get('claimed_subject_id'))
    except (ValueError, AttributeError):
        pass
    status = result.get('status') if isinstance(result, dict) else None
    verdict = result.get('verdict') if isinstance(result, dict) else None
    if status == 'OK' and type(verdict) is bool:
        headline, level, hint = ('TRUE', 'ok', 'Gait supports the claim.') if verdict else (
            'FALSE', 'fail', 'Gait does not match the claim.')
    elif status in RESULTS and verdict is None:
        headline, level, hint = RESULTS[status]
    else:
        headline, level, hint = 'UNRECOGNIZED RESULT', 'fail', 'Result is malformed; no decision shown.'
        status = status if isinstance(status, str) else '?'
    return dict(title='Verification result', headline=headline, level=level,
                rows=[('Claimed ID', claim, 'info'), ('Status', str(status)[:40], 'info')],
                footer=hint)


def render_text(screen, color=True, width=60):
    paint = (lambda text, level: '\x1b[{}m{}\x1b[0m'.format(ANSI[level], text)) if color else (
        lambda text, level: text)
    lines = [screen['title'][:width], '=' * min(width, 40), paint(screen['headline'][:width], screen['level']), '']
    for label, value, level in screen['rows']:
        line = '{:<14}{}'.format(label + ':', value)[:width]
        lines.append(line[:14] + paint(line[14:], level))
    return '\n'.join(lines + ['', screen['footer'][:width], ''])


def render_image(screen, width=800, height=480):
    import cv2
    import numpy as np
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)

    def put(text, y, scale, level, thick=2):
        cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, LEVELS[level], thick,
                    cv2.LINE_AA)
    put(screen['title'], int(height * .08), height / 700., 'info', 1)
    put(screen['headline'], int(height * .22), height / 330., screen['level'], 3)
    step = max(int(height * .075), 16)
    for index, (label, value, level) in enumerate(screen['rows']):
        put('{}: {}'.format(label, value)[:int(width / (height / 480. * 13))], int(height * .36) + step * index,
            height / 650., level, 1)
    put(screen['footer'], height - 14, height / 800., 'info', 1)
    return canvas


def save_png(screen, path, width=800, height=480):
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise ValueError('Refusing to overwrite an existing image')
    os.close(os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    if not cv2.imwrite(str(path), render_image(screen, width, height)):
        path.unlink()
        raise RuntimeError('Could not write image')
    return path


def show_window(screen, seconds=None, width=800, height=480):
    """Full-screen OpenCV window on the attached display (needs a desktop session)."""
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        raise RuntimeError('No DISPLAY: start a desktop session or use --render text/png')
    import cv2
    name = 'gait-status'
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name, render_image(screen, width, height))
    cv2.waitKey(0 if seconds is None else int(seconds * 1000))
    cv2.destroyWindow(name)

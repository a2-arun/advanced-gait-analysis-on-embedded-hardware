"""One-command Nano preflight. Read-only: never installs, upgrades, records or opens a camera.

Each check is PASS, FAIL, WARNING, NOT_TESTED or BLOCKED and names its scope: `host-independent`
checks mean the same on any machine; `requires-nano` checks are NOT_TESTED anywhere but on a
Jetson, so a workstation run can never look like hardware evidence.
"""
import ast
import glob
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .dbcheck import database_status
from .environment import collect_nano_environment, pinned_packages
from .model_slot import inventory

SCHEMA = 'nano-preflight-v1'
STATUSES = ('PASS', 'FAIL', 'WARNING', 'NOT_TESTED', 'BLOCKED')
ROOT = Path(__file__).resolve().parents[2]
# Operational guidance, not a validated specification.
MIN_RAM_MB, WARN_DISK_MB, MIN_DISK_MB = 3500, 5000, 1000
DISCLAIMER = ('This report checks the installed software and devices only. It does not prove '
              'biometric accuracy, model approval, real-time performance, or that a workstation '
              'run represents the Nano; NOT_TESTED means the check needs the original Nano.')


def run_text(command, timeout=10):
    try:
        done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return done.stdout.decode('utf8', 'replace') if done.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def nearest_existing(path):
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def read_meminfo_mb(path='/proc/meminfo'):
    try:
        match = re.search(r'MemTotal:\s+(\d+) kB', Path(path).read_text())
        return int(match.group(1)) // 1024 if match else None
    except OSError:
        return None


def display_probe():
    for status in sorted(glob.glob('/sys/class/drm/*HDMI*/status')):
        try:
            if Path(status).read_text().strip() == 'connected':
                return dict(connected=True, source=status)
        except OSError:
            pass
    if os.environ.get('DISPLAY'):
        return dict(connected=True, source='DISPLAY=' + os.environ['DISPLAY'])
    return dict(connected=False, source=None)


def tool_version(banner):
    line = (banner.splitlines() or [''])[0].strip()
    match = re.search(r'version (\S+)', line)
    return match.group(1) if match else line or None


def ffmpeg_probe():
    first = tool_version
    ffmpeg, ffprobe = (run_text([n, '-version']) if shutil.which(n) else '' for n in ('ffmpeg', 'ffprobe'))
    formats = run_text(['ffmpeg', '-hide_banner', '-formats']) if ffmpeg else ''
    encoders = run_text(['ffmpeg', '-hide_banner', '-encoders']) if ffmpeg else ''
    return dict(ffmpeg=first(ffmpeg), ffprobe=first(ffprobe), libx264='libx264' in encoders,
                v4l2=bool(re.search(r'\bv4l2\b', formats)))


def collect_facts(cfg, device='/dev/video0'):
    """Environment facts plus device probes. Camera nodes are stat'ed, never opened."""
    facts = collect_nano_environment(device)
    nodes = [dict(path=p, accessible=os.access(p, os.R_OK | os.W_OK))
             for p in sorted(glob.glob('/dev/video*'))]
    jetpack = run_text(['dpkg-query', '-W', '-f=${Version}', 'nvidia-jetpack']) \
        if shutil.which('dpkg-query') else ''
    facts['probes'] = dict(
        memory_mb=read_meminfo_mb(), display=display_probe(), video_nodes=nodes,
        disk_free_mb=shutil.disk_usage(str(nearest_existing(Path(cfg['database']).parent))).free // 2**20,
        ffmpeg=ffmpeg_probe(), v4l2_ctl=bool(shutil.which('v4l2-ctl')), jetpack_package=jetpack or None)
    return facts


def python36_grammar_errors(paths):
    bad = []
    for path in paths:
        try:
            source = Path(path).read_text()
            try:
                ast.parse(source, str(path), feature_version=(3, 6))
            except TypeError:  # interpreters before 3.8 are the target grammar already
                ast.parse(source, str(path))
        except (SyntaxError, ValueError, OSError):
            bad.append(str(path))
    return bad


def is_jetson(facts):
    return 'Jetson' in facts.get('hardware', '')


def nano_only(facts, observed, evaluate):
    if not is_jetson(facts):
        return 'NOT_TESTED', 'Not a Jetson ({} {}); observed here: {}'.format(
            facts.get('machine'), facts.get('hardware') or 'no device-tree model', observed)
    return evaluate()


def verdict(ok, good, bad):
    return ('PASS', good) if ok else ('FAIL', bad)


def board_check(facts):
    hardware = facts.get('hardware', '')
    if not is_jetson(facts):
        return nano_only(facts, hardware or 'n/a', None)
    if 'Jetson Nano' not in hardware or 'Orin' in hardware:
        return 'FAIL', 'Original Jetson Nano required, found: ' + hardware
    if '2GB' in hardware:
        return 'WARNING', hardware + ' has 2 GB RAM; the profile assumes about 4 GB'
    return 'PASS', hardware


def os_check(facts):
    def evaluate():
        problems = []
        if facts.get('ubuntu') != '18.04':
            problems.append('Ubuntu 18.04 required (found {})'.format(facts.get('ubuntu')))
        if not re.search(r'R32.*REVISION:\s*[67]\.', facts.get('l4t', '')):
            problems.append('L4T R32.6.x/R32.7.x (JetPack 4.6.x) required')
        pack = (facts.get('probes') or {}).get('jetpack_package')
        note = ' nvidia-jetpack package: {}.'.format(pack or 'not reported')
        if problems:
            return 'FAIL', '; '.join(problems) + '.' + note
        if pack and not pack.startswith('4.6'):
            return 'WARNING', 'L4T matches but nvidia-jetpack is ' + pack
        return 'PASS', 'Ubuntu 18.04, {}.{}'.format(facts['l4t'].splitlines()[0][:60], note)
    return nano_only(facts, '{} {}'.format(facts.get('ubuntu'), facts.get('l4t', '')[:40]), evaluate)


def torch_check(facts):
    version = facts.get('packages', {}).get('torch')
    return nano_only(facts, 'torch ' + str(version), lambda: verdict(
        bool(version) and str(version).startswith('1.10.0'),
        'torch ' + str(version), 'NVIDIA PyTorch 1.10.0 required (found {}; {})'.format(
            version, facts.get('import_errors', {}).get('torch', 'wrong version'))))


def cuda_check(facts):
    observed = 'cuda={} available={}'.format(facts.get('cuda'), facts.get('cuda_available'))
    return nano_only(facts, observed, lambda: verdict(
        facts.get('cuda') == '10.2' and bool(facts.get('cuda_available')), observed,
        'CUDA 10.2 must be usable from PyTorch; no CPU fallback (' + observed + ')'))


def memory_check(facts):
    mb = facts['probes'].get('memory_mb')
    return nano_only(facts, '{} MB'.format(mb), lambda: verdict(
        bool(mb) and mb >= MIN_RAM_MB, '{} MB total (shared CPU/GPU)'.format(mb),
        '{} MB is below the ~4 GB profile'.format(mb)))


def disk_check(facts):
    mb = facts['probes'].get('disk_free_mb')
    if mb is None or mb < MIN_DISK_MB:
        return 'FAIL', '{} MB free; at least {} MB needed'.format(mb, MIN_DISK_MB)
    if mb < WARN_DISK_MB:
        return 'WARNING', '{} MB free; {} MB recommended for recordings'.format(mb, WARN_DISK_MB)
    return 'PASS', '{} MB free'.format(mb)


def ffmpeg_check(facts):
    probe = facts['probes'].get('ffmpeg') or {}
    missing = [n for n, ok in (('ffmpeg', probe.get('ffmpeg')), ('libx264 encoder', probe.get('libx264')),
                               ('v4l2 input', probe.get('v4l2'))) if not ok]
    return verdict(not missing, 'ffmpeg {} with libx264 and v4l2'.format(probe.get('ffmpeg')),
                   'missing: ' + ', '.join(missing))


def ffprobe_check(facts):
    version = (facts['probes'].get('ffmpeg') or {}).get('ffprobe')
    return verdict(bool(version), 'ffprobe ' + str(version), 'ffprobe is missing')


def opencv_check(facts):
    version = facts.get('packages', {}).get('opencv')
    return nano_only(facts, 'opencv ' + str(version), lambda: verdict(
        version == '4.1.1', 'JetPack OpenCV ' + str(version),
        'JetPack system OpenCV 4.1.1 required (found {})'.format(version)))


def mediapipe_check(facts):
    version = facts.get('packages', {}).get('mediapipe')
    return nano_only(facts, 'mediapipe ' + str(version), lambda: verdict(
        version == '0.8.5', 'MediaPipe ' + str(version) + ' (pose assets: see `lab.py pose check`)',
        'MediaPipe 0.8.5 required (found {}; {})'.format(
            version, facts.get('import_errors', {}).get('mediapipe', 'wrong version'))))


def python_check(facts):
    version = facts.get('python', '')
    return nano_only(facts, 'python ' + version, lambda: verdict(
        version.startswith('3.6.'), 'Python ' + version, 'Python 3.6 required (found {})'.format(version)))


def camera_check(facts):
    nodes = facts['probes'].get('video_nodes', [])
    listing = ', '.join('{}({})'.format(n['path'], 'rw' if n['accessible'] else 'no access')
                        for n in nodes) or 'none'

    def evaluate():
        if not nodes:
            return 'FAIL', 'No /dev/video* node. Plug in the USB camera and re-run.'
        if not any(n['accessible'] for n in nodes):
            return 'FAIL', 'Camera nodes not readable/writable by this user: ' + listing
        return 'PASS', 'device node(s): {} (not opened; use `lab.py camera list`)'.format(listing)
    return nano_only(facts, listing, evaluate)


def display_check(facts):
    display = facts['probes'].get('display') or {}
    return nano_only(facts, str(display), lambda: (
        ('PASS', 'display detected via ' + str(display['source'])) if display.get('connected') else
        ('WARNING', 'No HDMI display or DISPLAY detected; headless runs still write reports')))


def package_check(facts):
    packages = facts.get('packages', {})
    pins = pinned_packages()
    wrong = ['{} {} != {}'.format(n, packages.get(n), v) for n, v in sorted(pins.items())
             if packages.get(n) != v]
    return nano_only(facts, 'numpy {}'.format(packages.get('numpy')), lambda: verdict(
        not wrong, 'all {} pinned packages match requirements-jetson.txt'.format(len(pins)),
        'pin mismatch: ' + '; '.join(wrong)))


def database_check(cfg):
    path = Path(cfg['database'])
    if path.is_symlink():
        return 'FAIL', 'Database path is a symlink'
    status = database_status(path)
    state = status['state']
    if state in ('CORRUPT', 'NEWER_THAN_SOFTWARE'):
        return 'FAIL', 'database state ' + state
    folder = nearest_existing(path.parent)
    if not os.access(str(folder), os.W_OK | os.X_OK):
        return 'FAIL', 'cannot create or write in ' + str(folder)
    if state in ('NEEDS_MIGRATION', 'UNINITIALIZED'):
        return 'WARNING', 'database state {} (run `lab.py db init`)'.format(state)
    if state == 'ABSENT':
        return 'PASS', 'writable location; database not initialized yet (`lab.py db init`)'
    return 'PASS', 'schema v{} READY; {} references, {} subjects'.format(
        status['schema_version'], status['references'], status['subjects'])


def model_check(cfg, facts):
    bundle = cfg.get('bundle_dir')
    state = inventory(cfg['model_slot_dir'], facts, cfg['source_dir'],
                      baseline_bundle=bundle if bundle and Path(bundle).is_dir() else None)
    if state['state'] == 'ACTIVE':
        return 'PASS', 'active model package: ' + state['active']
    parts = ['MODEL_NOT_DEPLOYED: no activated final model.']
    for package in state['packages']:
        open_stages = ['{}={}'.format(n, s['status']) for n, s in package['stages'].items()
                       if s['status'] != 'PASS' and n != 'active']
        parts.append('{}: {}.'.format(package['package_id'], ', '.join(open_stages)
                                      if open_stages else 'qualified but not activated'))
    if state['development_baseline']:
        parts.append('A development baseline bundle is present; it is provisional GaitPT, '
                     'not deployed and never the final model.')
    return 'BLOCKED', ' '.join(parts)


CHECKS = (  # id, title, scope, function
    ('board', 'Board identification', 'requires-nano', lambda c, f: board_check(f)),
    ('os_l4t_jetpack', 'Ubuntu / L4T / JetPack', 'requires-nano', lambda c, f: os_check(f)),
    ('python', 'Python version', 'requires-nano', lambda c, f: python_check(f)),
    ('cuda', 'CUDA available to PyTorch', 'requires-nano', lambda c, f: cuda_check(f)),
    ('pytorch', 'PyTorch', 'requires-nano', lambda c, f: torch_check(f)),
    ('memory', 'System memory', 'requires-nano', lambda c, f: memory_check(f)),
    ('disk', 'Free disk space', 'host-independent', lambda c, f: disk_check(f)),
    ('ffmpeg', 'FFmpeg (libx264, v4l2)', 'host-independent', lambda c, f: ffmpeg_check(f)),
    ('ffprobe', 'FFprobe', 'host-independent', lambda c, f: ffprobe_check(f)),
    ('opencv', 'OpenCV (JetPack)', 'requires-nano', lambda c, f: opencv_check(f)),
    ('mediapipe', 'MediaPipe', 'requires-nano', lambda c, f: mediapipe_check(f)),
    ('usb_camera', 'USB camera discovery', 'requires-nano', lambda c, f: camera_check(f)),
    ('display', 'HDMI / display', 'requires-nano', lambda c, f: display_check(f)),
    ('database_dir', 'Database directory', 'host-independent', lambda c, f: database_check(c)),
    ('model_artifacts', 'Model artifacts', 'host-independent', lambda c, f: model_check(c, f)),
    ('package_compatibility', 'Pinned package compatibility', 'requires-nano',
     lambda c, f: package_check(f)),
    ('source_python36_grammar', 'Source parses as Python 3.6 grammar', 'host-independent',
     lambda c, f: (lambda bad: verdict(not bad, 'src/verification and scripts parse',
                                       'does not parse: ' + ', '.join(bad)))(python36_grammar_errors(
         sorted((ROOT / 'src/verification').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))))),
)
MODEL_IDS = ('model_artifacts',)


def overall(statuses):
    if 'FAIL' in statuses:
        return 'FAIL'
    if 'NOT_TESTED' in statuses or 'BLOCKED' in statuses:
        return 'INCOMPLETE' if statuses != ['BLOCKED'] else 'BLOCKED'
    return 'WARNING' if 'WARNING' in statuses else 'PASS'


def run_preflight(cfg, facts):
    checks = []
    for check_id, title, scope, function in CHECKS:
        try:
            status, detail = function(cfg, facts)
        except Exception as exc:  # a broken probe is a result, not a crash
            status, detail = 'FAIL', 'check raised {}: {}'.format(type(exc).__name__, exc)
        checks.append(dict(id=check_id, title=title, scope=scope, status=status, detail=detail))
    hardware = [c['status'] for c in checks if c['id'] not in MODEL_IDS]
    model = [c['status'] for c in checks if c['id'] in MODEL_IDS]
    counts = dict((s, sum(1 for c in checks if c['status'] == s)) for s in STATUSES)
    return dict(schema=SCHEMA, generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                host=dict(machine=facts.get('machine'), python=facts.get('python'),
                          hardware=facts.get('hardware') or None, on_nano=nano_hardware(facts)),
                checks=checks, disclaimer=DISCLAIMER,
                summary=dict(counts=counts, hardware_overall=overall(hardware),
                             model_overall=overall(model),
                             overall=overall([c['status'] for c in checks])))


def nano_hardware(facts):
    return 'Jetson Nano' in facts.get('hardware', '') and 'Orin' not in facts.get('hardware', '')


def markdown(report):
    lines = ['# Jetson Nano preflight', '',
             'Generated {} on {} ({}), Python {}. On original Nano: {}.'.format(
                 report['generated_at'], report['host']['hardware'] or 'a non-Jetson host',
                 report['host']['machine'], report['host']['python'], report['host']['on_nano']),
             '', 'Hardware: **{hardware_overall}**; model: **{model_overall}**; overall: '
             '**{overall}**.'.format(**report['summary']), '',
             '| id | scope | status | detail |', '|---|---|---|---|']
    for c in report['checks']:
        lines.append('| {} | {} | {} | {} |'.format(c['id'], c['scope'], c['status'],
                                                    c['detail'].replace('|', '/')))
    return '\n'.join(lines + ['', report['disclaimer'], ''])


def write_reports(report, directory):
    import json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stem = 'preflight-{}-{}'.format(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                                    uuid.uuid4().hex[:6])
    paths = []
    for suffix, text in (('.json', json.dumps(report, indent=2) + '\n'), ('.md', markdown(report))):
        path = directory / (stem + suffix)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        paths.append(path)
    return tuple(paths)

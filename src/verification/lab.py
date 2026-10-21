"""Hardware-lab commands that need no biometric model: preflight, camera, pose, db, model slot,
display. Every command is explicit; nothing here enrolls, verifies or invents a decision."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import (camera_diag, dbcheck, display, model_slot, pose_diag, readiness)
from .cli import configuration


def emit(value):
    print(json.dumps(value, indent=2))


def facts_provider(cfg):
    cache = []

    def provide():
        if not cache:
            cache.append(readiness.collect_facts(cfg))
        return cache[0]
    return provide


def baseline_bundle(cfg):
    return cfg['bundle_dir'] if Path(cfg['bundle_dir']).is_dir() else None


def package_encoder(cfg, package):
    """`encode(crops)` of an installed package, through the backend its contract names."""
    contract = model_slot.read_json(Path(package) / 'contract.json')
    backend = model_slot.backend_for(contract)
    return backend(cfg['source_dir'], str(package), 'parity-only', cfg['device']).encode_batch


def cmd_preflight(args, cfg):
    report = readiness.run_preflight(cfg, readiness.collect_facts(cfg, args.device))
    json_path, md_path = readiness.write_reports(report, cfg['reports_dir'])
    if args.json:
        emit(report)
    else:
        print(readiness.markdown(report))
        print('Saved: {}\n       {}'.format(json_path, md_path))
    return 2 if report['summary']['overall'] == 'FAIL' else 0


def cmd_camera(args, cfg):
    if args.action == 'list':
        emit(camera_diag.list_cameras())
        return 0
    if args.action == 'analyze':
        report = camera_diag.analyze_video(args.video)
        emit(report)
        return 2 if report['status'] == 'FAIL' else 0
    resolution = None if args.width is None and args.height is None else [args.width, args.height]
    if resolution and None in resolution:
        raise ValueError('Give both --width and --height')
    report = camera_diag.record_diagnostic(cfg, args.seconds, args.device, resolution, args.fps)
    emit(report)
    return 2 if report['analysis']['status'] == 'FAIL' else 0


def cmd_pose(args, cfg):
    if args.action == 'check':
        result = pose_diag.check_named_backend(args.backend)
        emit(result)
        return 0 if result['status'] == 'PASS' else 2
    if not Path(args.video).is_file():
        raise ValueError('--video must be an existing recording chosen by the operator')
    backend = pose_diag.load_backend(args.backend)
    report = pose_diag.extract_pose_report(args.video, backend, args.overlay)
    folder = Path(cfg['diagnostics_dir']) / 'pose'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / 'pose-diag-{}.json'.format(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    camera_diag.write_private_json(target, report)
    report['report'] = str(target)
    emit(report)
    return 0


def cmd_db(args, cfg):
    status = (dbcheck.initialize_database if args.action == 'init' else dbcheck.database_status)(
        cfg['database'])
    emit(status)
    return 0 if status['state'] in ('READY', 'ABSENT') else 2


def cmd_model(args, cfg):
    slot, provide = cfg['model_slot_dir'], facts_provider(cfg)
    if args.action == 'status':
        emit(model_slot.inventory(slot, provide(), cfg['source_dir'], baseline_bundle(cfg)))
    elif args.action == 'inspect':
        emit(model_slot.inspect_package(model_slot.package_path(slot, args.package), provide(),
                                        cfg['source_dir']))
    elif args.action == 'install':
        emit(dict(installed=str(model_slot.install_package(args.package, slot)), active=False))
    elif args.action == 'parity':
        directory = model_slot.package_path(slot, args.package)
        emit(model_slot.record_parity(directory, args.fixture, package_encoder(cfg, directory), provide()))
    elif args.action == 'activate':
        emit(model_slot.activate(slot, args.package, provide(), cfg['source_dir']))
    else:
        model_slot.deactivate(slot)
        emit(dict(active=None))
    return 0


def cmd_display(args, cfg):
    provide = facts_provider(cfg)
    if args.action == 'result':
        path = Path(args.event) if args.event else max(
            Path(cfg['events_dir']).glob('*.json'), key=lambda p: p.stat().st_mtime, default=None)
        if path is None:
            raise ValueError('No verification event found')
        def build():
            return display.result_screen(json.loads(Path(path).read_text()))
    else:
        def build():
            return display.status_screen(display.gather_state(cfg, provide))
    for tick in range(args.iterations if args.watch else 1):
        screen = build()
        if args.render == 'text':
            print(display.render_text(screen, color=not args.no_color and sys.stdout.isatty()))
        elif args.render == 'png':
            if not args.output:
                raise ValueError('--render png needs --output')
            display.save_png(screen, args.output)
        else:
            display.show_window(screen, seconds=args.watch or 5)
        if args.watch and tick + 1 < args.iterations:
            time.sleep(args.watch)
    return 0


def required(subparsers):
    subparsers.required = True  # the required= keyword only exists from Python 3.7
    return subparsers


def parser():
    top = argparse.ArgumentParser(prog='lab.py', description=__doc__)
    top.add_argument('--config')
    commands = top.add_subparsers(dest='command')
    pre = commands.add_parser('preflight', help='Read-only Nano readiness report (JSON + Markdown)')
    pre.add_argument('--device', default='/dev/video0')
    pre.add_argument('--json', action='store_true')
    cam = commands.add_parser('camera', help='Model-free USB camera diagnostics')
    sub = required(cam.add_subparsers(dest='action'))
    sub.add_parser('list')
    rec = sub.add_parser('record', help='Operator-triggered short diagnostic clip (kept apart from originals)')
    rec.add_argument('--seconds', type=int, default=5)
    rec.add_argument('--device')
    rec.add_argument('--width', type=int)
    rec.add_argument('--height', type=int)
    rec.add_argument('--fps', type=int)
    sub.add_parser('analyze').add_argument('--video', required=True)
    pose = commands.add_parser('pose', help='Independent pose-estimator diagnostics')
    sub = required(pose.add_subparsers(dest='action'))
    sub.add_parser('check').add_argument('--backend', default=pose_diag.MEDIAPIPE_LITE)
    ext = sub.add_parser('extract')
    ext.add_argument('--video', required=True)
    ext.add_argument('--backend', default=pose_diag.MEDIAPIPE_LITE)
    ext.add_argument('--overlay', help='Write a new skeleton overlay video here')
    db = commands.add_parser('db', help='Verification database readiness')
    db.add_argument('action', choices=['init', 'check'])
    model = commands.add_parser('model', help='Final-model slot: status, install, parity, activate')
    sub = required(model.add_subparsers(dest='action'))
    sub.add_parser('status')
    sub.add_parser('deactivate')
    for name in ('inspect', 'install', 'parity', 'activate'):
        item = sub.add_parser(name)
        item.add_argument('package', help='package id' if name != 'install' else 'package directory')
        if name == 'parity':
            item.add_argument('--fixture', required=True)
    show = commands.add_parser('display', help='HDMI/LCD status or result screen')
    sub = required(show.add_subparsers(dest='action'))
    for name in ('status', 'result'):
        item = sub.add_parser(name)
        item.add_argument('--render', choices=['text', 'png', 'window'], default='text')
        item.add_argument('--output')
        item.add_argument('--no-color', action='store_true')
        item.add_argument('--watch', type=float, help='Redraw every N seconds')
        item.add_argument('--iterations', type=int, default=1000000, help=argparse.SUPPRESS)
        if name == 'result':
            item.add_argument('--event', help='Verification event JSON (default: newest in events_dir)')
    return top


HANDLERS = dict(preflight=cmd_preflight, camera=cmd_camera, pose=cmd_pose, db=cmd_db,
                model=cmd_model, display=cmd_display)


def main(argv=None):
    top = parser()
    args = top.parse_args(argv)
    if not args.command:
        top.print_help()
        return 2
    try:
        return HANDLERS[args.command](args, configuration(args.config))
    except Exception as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        return 2

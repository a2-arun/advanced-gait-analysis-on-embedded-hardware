"""Batch capture/enrollment/verification, executed entirely on the local device."""
import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .contract import InsufficientGait, identifier
from .registry import RecordingRegistry, private_json
from .service import Verifier, response
from .store import EnrollmentStore

ROOT = Path(__file__).resolve().parents[2]


def configuration(path):
    import yaml
    path = Path(path or ROOT / 'config.yaml').resolve()
    value = yaml.safe_load(path.read_text())
    if value['schema_version'] != 2:
        raise ValueError('Use verification config schema version 2')
    value.setdefault('capture_dir', 'data/raw/camera')
    for key in ('source_dir', 'bundle_dir', 'database', 'recordings', 'events_dir', 'log_file', 'capture_dir'):
        value[key] = str((path.parent / value[key]).resolve())
    if value.get('policy'):
        value['policy'] = str((path.parent / value['policy']).resolve())
    return value


def operator_logger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ValueError('Linked log path refused')
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(fd)
    os.chmod(str(path), 0o600)
    logger = logging.getLogger('gait.verification')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.addHandler(logging.FileHandler(str(path)))
    return logger


def emit_verification(cfg, result):
    private_json(Path(cfg['events_dir']) / (result['request_id'] + '.json'), result)
    print(json.dumps(result))
    return 0 if result['status'] == 'OK' else 2


def protect_originals(cfg, registry, extra_sources=()):
    sources = [row['path'] for row in registry.rows()] + list(extra_sources)
    for original in sources:
        source = Path(original)
        for key in ('database', 'log_file', 'recordings'):
            target = Path(cfg[key])
            if (target.resolve() == source.resolve()
                    or (target.exists() and source.exists() and target.samefile(source))):
                raise ValueError('Mutable artifact path aliases an immutable original')


def consent_reference(args):
    if args.consent_ref:
        return args.consent_ref
    if args.consent:
        return 'operator-attestation:' + datetime.now(timezone.utc).isoformat(timespec='seconds')
    raise ValueError('New recordings require --consent or --consent-ref')


def existing_recording(args, registry):
    try:
        return registry.resolve(args.video)
    except (ValueError, OSError, KeyError, TypeError):
        if not args.register_original:
            raise ValueError('Register the original first or explicitly use --register-original')
        return registry.register_operational(args.video,
            args.subject_id if args.command == 'enroll' else None, consent_reference(args),
            'enrollment' if args.command == 'enroll' else 'query', args.captured_at)


def camera_recording(args, cfg, registry):
    from .capture import capture_video
    consent = consent_reference(args)
    captured_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    video = Path(cfg['capture_dir']) / ('walk_' + uuid.uuid4().hex + '.mp4')
    capture_video(video, cfg.get('camera', {}), args.seconds, args.device)
    row = registry.register_operational(video,
        args.subject_id if args.command == 'enroll' else None, consent,
        'enrollment' if args.command == 'enroll' else 'query', captured_at)
    args.video = str(video)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local claimed-identity gait verification')
    commands = parser.add_subparsers(dest='command')
    for command in ('check', 'recipe', 'enroll', 'verify', 'register', 'capture'):
        sub = commands.add_parser(command)
        sub.add_argument('--config')
        if command in ('enroll', 'verify'):
            sub.add_argument('--video', help='Use an original video instead of recording the USB camera')
            sub.add_argument('--seconds', type=int, default=30)
            sub.add_argument('--device', help='USB camera device; defaults to config camera.device_id')
            sub.add_argument('--consent', action='store_true', help='Attest consent for this new recording')
            sub.add_argument('--consent-ref', help='Existing consent reference for this new recording')
            sub.add_argument('--register-original', action='store_true',
                             help='Explicitly register --video as a new operational original, never a derivative')
            sub.add_argument('--captured-at', help='Actual original capture time, if known, with UTC offset')
        if command == 'register':
            sub.add_argument('--video', required=True)
            sub.add_argument('--subject-id', required=True)
        if command == 'enroll':
            sub.add_argument('--subject-id', '--name', dest='subject_id', required=True)
        if command == 'verify':
            sub.add_argument('--claimed-id', '--identity', dest='claimed_id', required=True)
            sub.add_argument('--private-diagnostics-dir',
                             help='Opt-in private score artifacts for authorized offline evaluation')
        if command in ('enroll', 'verify'):
            sub.add_argument('--single-walker-reviewed', action='store_true', required=True,
                             help='Attest a reviewed video or supervised camera scene has one walker and no bystanders')
        if command == 'register':
            sub.add_argument('--take-id', required=True)
            sub.add_argument('--session-id', required=True)
            sub.add_argument('--view', required=True, choices=['left45', 'right45', 'toward', 'away'])
            sub.add_argument('--role', required=True, choices=['development', 'calibration'])
            sub.add_argument('--consent-ref', required=True)
            sub.add_argument('--captured-at', required=True, help='Actual capture time, ISO 8601 with offset')
        if command == 'check':
            sub.add_argument('--require-nano', action='store_true')
        if command == 'capture':
            sub.add_argument('--output', required=True)
            sub.add_argument('--device', default='/dev/video0')
            sub.add_argument('--seconds', type=int, default=30)
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    logger = None
    try:
        cfg = configuration(args.config)
        registry = RecordingRegistry(cfg['recordings'])
        protect_originals(cfg, registry, [args.video] if getattr(args, 'video', None) else [])
        if args.command == 'capture':
            from .capture import capture_video
            target = Path(args.output)
            if any(target.resolve() == Path(row['path']).resolve() for row in registry.rows()):
                raise ValueError('Output path belongs to a registered original')
            capture_video(target, cfg.get('camera', {}), args.seconds, args.device)
            print(json.dumps({'status': 'CAPTURED', 'path': str(target)}))
            return 0
        if args.command == 'register':
            registry.register(args.video, args.take_id, args.subject_id, args.session_id,
                              args.view, args.role, args.consent_ref, args.captured_at)
            print(json.dumps({'status': 'REGISTERED', 'take_id': args.take_id}))
            return 0
        logger = operator_logger(cfg['log_file'])
        store = None
        if args.command in ('verify', 'enroll'):
            try:
                identifier(getattr(args, 'claimed_id', getattr(args, 'subject_id', None)))
                if args.video and not Path(args.video).is_file():
                    raise ValueError('Video does not exist')
                if not args.video:
                    consent_reference(args)
            except (ValueError, OSError, KeyError, TypeError):
                claim = getattr(args, 'claimed_id', getattr(args, 'subject_id', None))
                result = response(claim, 'INVALID_INPUT')
                if args.command == 'verify':
                    return emit_verification(cfg, result)
                print(json.dumps(result))
                return 2
            store = EnrollmentStore(cfg['database'])
            if args.command == 'verify' and not store.has_claim(args.claimed_id):
                return emit_verification(cfg, response(args.claimed_id, 'CLAIM_NOT_ENROLLED'))
            if args.video:
                try:
                    row = existing_recording(args, registry)
                    if args.command == 'enroll' and row['subject_id'] != args.subject_id:
                        raise ValueError('Enrollment owner differs from registered owner')
                except (ValueError, OSError, KeyError, TypeError):
                    result = response(getattr(args, 'claimed_id', getattr(args, 'subject_id', None)),
                                      'INVALID_INPUT')
                    if args.command == 'verify':
                        return emit_verification(cfg, result)
                    print(json.dumps(result))
                    return 2
        from .backend import LocalGaitPT, runtime_recipe
        recipe, details = runtime_recipe(cfg['source_dir'], cfg['bundle_dir'], cfg['device'])
        if args.command in ('recipe', 'check'):
            import torch
            if args.command == 'check':
                if args.require_nano:
                    model = Path('/proc/device-tree/model').read_text().strip('\x00\n')
                    if 'Jetson Nano' not in model or 'Orin' in model:
                        raise RuntimeError('Original Jetson Nano hardware required')
                encoder = LocalGaitPT(cfg['source_dir'], cfg['bundle_dir'], recipe, cfg['device'])
                import numpy as np
                batch = np.random.RandomState(7).uniform(0, 1, (2, 30, 18, 2)).astype(np.float32)
                a, b = encoder.encode_batch(batch), encoder.encode_batch(batch)
                if not np.allclose(a, b, atol=1e-5):
                    raise RuntimeError('Frozen inference repeatability check failed')
                from .backend import pose_runtime_smoke
                pose_runtime_smoke()
                details['measured_embedding_dim'] = encoder.embedding_dim
                details['pose_graph_smoke'] = True
                details['cuda_available'] = torch.cuda.is_available()
                details['engineering_smoke_only'] = True
            print(json.dumps({'recipe': recipe, 'details': details}, indent=2))
            return 0
        encoder = None
        def factory():
            nonlocal encoder
            if encoder is None:
                encoder = LocalGaitPT(cfg['source_dir'], cfg['bundle_dir'], recipe, cfg['device'])
            return encoder
        if args.command == 'verify' and not store.claim(args.claimed_id, recipe):
            stale = store.incompatible_count(args.claimed_id, recipe)
            if stale:
                logger.warning('Claim has %d reference(s) from another or legacy recipe; '
                               're-enroll or reprocess the preserved originals', stale)
            return emit_verification(cfg, response(args.claimed_id, 'CLAIM_NOT_ENROLLED'))
        if not args.video:
            factory()  # Reject broken weights/CUDA before asking the user to walk.
            from .backend import pose_runtime_smoke
            pose_runtime_smoke()
            row = camera_recording(args, cfg, registry)
        if args.command == 'verify':
            def diagnostics(record):
                if args.private_diagnostics_dir:
                    directory = Path(args.private_diagnostics_dir).resolve()
                    for source in registry.rows():
                        if directory == Path(source['path']).parent:
                            raise ValueError('Diagnostics must be outside original recording folders')
                    private_json(directory / (record['request_id'] + '.json'), record)
            result = Verifier(store, registry, recipe, factory, cfg.get('policy'), logger,
                              diagnostics if args.private_diagnostics_dir else None).verify(
                args.video, args.claimed_id)
            return emit_verification(cfg, result)
        embedding = factory().encode(args.video)
        if registry.resolve(args.video) != row:
            raise ValueError('Original changed during enrollment')
        store.add(args.subject_id, row['original_take_id'], row['sha256'], recipe,
                  embedding, row['role'], details.get('runtime_fingerprint'))
        print(json.dumps({'status': 'ENROLLED', 'subject_id': args.subject_id,
                          'take_id': row['original_take_id'], 'recipe': recipe,
                          'database': cfg['database'],
                          'reference_count': len(store.claim(args.subject_id, recipe))}))
        return 0
    except Exception as exc:
        if logger:
            logger.exception('Local command failed')
        else:
            print('Setup failed: ' + str(exc), file=sys.stderr)
        if args.command in ('verify', 'enroll'):
            status = 'INSUFFICIENT_GAIT' if isinstance(exc, InsufficientGait) else 'PROCESSING_ERROR'
            print(json.dumps(response(getattr(args, 'claimed_id', getattr(args, 'subject_id', None)), status)))
        else:
            print('ERROR: ' + str(exc), file=sys.stderr)
        return 2

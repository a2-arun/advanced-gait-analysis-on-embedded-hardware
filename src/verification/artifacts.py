"""Restricted workstation export; pickle-free Nano load. Never imports legacy weights."""
import json
import io
import os
import struct
import zipfile
from pathlib import Path

import numpy as np

from .contract import sha256_file

REVISION = '4977618a36ef77ee9a2863c9f8e27ab862894d70'
CHECKPOINT_SHA = 'e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637'
SOURCE_SHA = '2f2e3f7acfc68756b464c97dbd208062215291e7680218b2f43a8df3bcd419f3'
CONFIG_SHA = 'ccdf1aa7c840547b359740399109210c2e2cbf9e355e29290ee8f48675db321e'


def validate_tensor_header(stream, file_size):
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError('Unsupported NPY version')
    size = 2 if version == (1, 0) else 4
    length_bytes = stream.read(size)
    if len(length_bytes) != size:
        raise ValueError('Truncated tensor header')
    length = struct.unpack('<H' if size == 2 else '<I', length_bytes)[0]
    if length > 10000:
        raise ValueError('Oversized tensor header')
    reader = (np.lib.format.read_array_header_1_0 if size == 2
              else np.lib.format.read_array_header_2_0)
    shape, _, dtype = reader(io.BytesIO(length_bytes + stream.read(length)))
    if dtype != np.float32 or len(shape) > 8:
        raise ValueError('Expected bounded float32 tensor')
    data_size = 4
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError('Invalid tensor shape')
        data_size *= dimension
        if data_size > 32*1024*1024:
            raise ValueError('Oversized declared tensor shape')
    if file_size != 8 + size + length + data_size:
        raise ValueError('Tensor header and payload size disagree')


def load_arrays(path):
    with zipfile.ZipFile(str(path)) as archive:
        entries = archive.infolist()
        if (len(entries) > 2000 or sum(i.file_size for i in entries) > 32*1024*1024
                or len(set(i.filename for i in entries)) != len(entries)):
            raise ValueError('Oversized or duplicate weight archive')
        for entry in entries:
            if not entry.filename.endswith('.npy'):
                raise ValueError('Non-tensor archive entry')
            with archive.open(entry) as stream:
                validate_tensor_header(stream, entry.file_size)
    result = {}
    with np.load(str(path), allow_pickle=False) as archive:
        for name in archive.files:
            array = archive[name]
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('Only finite float32 tensor arrays are supported')
            result[name] = array.copy()
    if not result:
        raise ValueError('Empty model bundle')
    return result


def prepare_bundle(checkpoint, destination):
    if sha256_file(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Requires the audited official GREW checkpoint SHA-256')
    import torch
    # Old torch must fail here rather than falling back to unrestricted pickle.
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    arrays = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not key.startswith('module.'):
            raise ValueError('Unexpected official checkpoint format')
        arrays[key[len('module.'):]] = tensor.detach().cpu().numpy()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = destination / 'weights.npz'
    np.savez(str(path), **arrays)
    os.chmod(str(path), 0o600)
    load_arrays(path)
    manifest = dict(schema='gaitpt-npz-v1', checkpoint_sha256=CHECKPOINT_SHA,
                    source_revision=REVISION, source_sha256=SOURCE_SHA,
                    config_sha256=CONFIG_SHA, weights_sha256=sha256_file(path))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def check_bundle(source, bundle):
    source, bundle = Path(source), Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for key, value in [('schema', 'gaitpt-npz-v1'), ('checkpoint_sha256', CHECKPOINT_SHA),
                       ('source_revision', REVISION), ('source_sha256', SOURCE_SHA),
                       ('config_sha256', CONFIG_SHA)]:
        if manifest.get(key) != value:
            raise ValueError('Incompatible model manifest: ' + key)
    if (sha256_file(source / 'gaitpt.py') != SOURCE_SHA
            or sha256_file(source / 'configs/gaitpt_config.yaml') != CONFIG_SHA
            or sha256_file(bundle / 'weights.npz') != manifest['weights_sha256']):
        raise ValueError('Source/config/weight integrity check failed')
    return manifest

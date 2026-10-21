"""Versioned, model-agnostic deployment contract.

The contract names everything that changes what an embedding MEANS (model, pose source,
preprocessing, aggregation, similarity). Its semantic identity is the enrollment/policy
binding. Runtime and host data (torch/CUDA/ffmpeg/python/device/paths) is provenance only
and never part of that identity. The contract carries no business logic.
"""
import re

import numpy as np

from .artifacts import CHECKPOINT_SHA, CONFIG_SHA, REVISION, SOURCE_SHA
from .contract import GAITPT_ORDER, MP_COCO, QUALITY_RECIPE, fingerprint

SCHEMA = 'nano-deployment-contract-v1'
REFERENCE_AGGREGATION = 'normalized-equal-weight-centroid-of-unit-walk-embeddings-v1'
SIMILARITY = 'cosine'
GAITPT_INPUT_SHAPE = (2, 30, 18, 2)
GAITPT_EMBEDDING_DIM = 256  # research configs/gait_verification/models.json: output_dimension

# Every semantic section lists its exact keys: unknown or missing keys are refused.
SECTIONS = {
    'model': ('family', 'variant', 'architecture_sha256', 'source_revision', 'source_sha256',
              'checkpoint_sha256', 'weights_format', 'weights_sha256'),
    'pose': ('estimator', 'version', 'settings', 'asset_sha256', 'layout'),
    'preprocessing': ('coordinate_system', 'normalization', 'joint_mapping', 'joint_order',
                      'temporal', 'quality', 'input_shape'),
    'embedding': ('dim', 'normalization', 'window_aggregation', 'walk_aggregation'),
    'scoring': ('reference_aggregation', 'similarity'),
    'numeric': ('precision',),
}
# Informational sections: validated as objects, excluded from the semantic identity.
PROVENANCE = ('runtime_requirements', 'parity', 'provenance')
HASHES = (('model', 'architecture_sha256'), ('model', 'source_sha256'),
          ('model', 'checkpoint_sha256'), ('model', 'weights_sha256'))


class ContractError(ValueError):
    pass


class UnsupportedModel(ContractError):
    pass


def validate_contract(contract):
    """Machine-check a contract; returns it unchanged or raises ContractError."""
    if not isinstance(contract, dict) or contract.get('schema') != SCHEMA:
        raise ContractError('Unknown deployment contract schema')
    allowed = set(SECTIONS) | set(PROVENANCE) | {'schema'}
    if set(contract) - allowed or set(SECTIONS) - set(contract):
        raise ContractError('Deployment contract sections differ from ' + SCHEMA)
    for name, keys in SECTIONS.items():
        section = contract[name]
        if not isinstance(section, dict) or set(section) != set(keys):
            raise ContractError('Deployment contract section keys differ: ' + name)
    for name in PROVENANCE:
        if name in contract and not isinstance(contract[name], dict):
            raise ContractError('Provenance section must be an object: ' + name)
    for section, key in HASHES:
        if not re.fullmatch('[0-9a-f]{64}', str(contract[section][key])):
            raise ContractError('Expected SHA-256 for ' + section + '.' + key)
    model, shape = contract['model'], contract['preprocessing']['input_shape']
    if not re.fullmatch('[a-z0-9_]{1,32}', str(model['family'])) or not model['variant']:
        raise ContractError('Malformed model family/variant')
    if model['weights_format'] != 'npz-float32-v1':
        raise ContractError('Unsupported weight serialization (pickle is never accepted)')
    if (not isinstance(shape, list) or not shape
            or not all(type(x) is int and x > 0 for x in shape)):
        raise ContractError('Input shape must be a list of positive integers')
    dim = contract['embedding']['dim']
    if type(dim) is not int or dim < 1:
        raise ContractError('Embedding dimension must be a positive integer')
    if contract['numeric']['precision'] != 'float32':
        raise ContractError('Only float32 inference is contracted')
    return contract


def semantic_id(contract):
    """Digest of the biometric meaning only; this is what enrollments/policies bind to."""
    validate_contract(contract)
    return fingerprint({name: contract[name] for name in sorted(SECTIONS)})


def gaitpt_grew_contract(manifest, pose):
    """The deployed frozen GaitPT-GREW contract. `pose` = version/asset_sha256/settings."""
    return validate_contract(dict(
        schema=SCHEMA,
        model=dict(family='gaitpt', variant='grew-official', architecture_sha256=CONFIG_SHA,
                   source_revision=REVISION, source_sha256=SOURCE_SHA,
                   checkpoint_sha256=CHECKPOINT_SHA, weights_format='npz-float32-v1',
                   weights_sha256=manifest['weights_sha256']),
        pose=dict(estimator='mediapipe-solutions-single-person-lite', version=pose['version'],
                  settings=pose['settings'], asset_sha256=list(pose['asset_sha256']),
                  layout='mediapipe33'),
        preprocessing=dict(coordinate_system='display-oriented-image-pixels-x-right-y-down',
                           normalization='divide-both-axes-by-image-height-no-centering',
                           joint_mapping=list(MP_COCO), joint_order=list(GAITPT_ORDER),
                           temporal='grew-two-center30-height-v1:longest-contiguous-segment',
                           quality=QUALITY_RECIPE, input_shape=list(GAITPT_INPUT_SHAPE)),
        embedding=dict(dim=GAITPT_EMBEDDING_DIM, normalization='l2-per-crop-mean-l2',
                       window_aggregation='none:single-longest-segment',
                       walk_aggregation='normalize-crops-mean-normalize-walk-centroid-v1'),
        scoring=dict(reference_aggregation=REFERENCE_AGGREGATION, similarity=SIMILARITY),
        numeric=dict(precision='float32')))


def semantic_identity(contract, implementation):
    """Enrollment/policy binding. `implementation` = hashes of the pixel->embedding code."""
    return fingerprint(dict(contract=semantic_id(contract), implementation=implementation))


def check_model_input(batch, shape):
    if batch.shape != tuple(shape) or batch.dtype != np.float32 or not np.isfinite(batch).all():
        raise ValueError('Invalid model input: expected finite float32 ' + str(tuple(shape)))
    return batch


def check_embedding(vector, dim):
    if len(vector) != dim:
        raise ValueError('Embedding dimension differs from the deployment contract')
    return vector

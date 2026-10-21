import copy
import numpy as np
import pytest

from src.verification.backend import LocalGaitPT, backend_for
from src.verification.contract import GAITPT_ORDER, MP_COCO
from src.verification.deployment import (ContractError, UnsupportedModel, check_embedding,
                                         check_model_input, gaitpt_grew_contract,
                                         semantic_id, semantic_identity, validate_contract)

IMPL = {'backend': 'a' * 64, 'media': 'b' * 64}


def contract(**pose):
    values = dict(version='0.8.5', settings={'model_complexity': 0}, asset_sha256=['c' * 64])
    values.update(pose)
    return gaitpt_grew_contract({'weights_sha256': 'd' * 64}, values)


def test_gaitpt_contract_is_valid_and_pins_the_audited_pipeline():
    value = contract()
    assert validate_contract(value) is value
    assert value['preprocessing']['joint_mapping'] == list(MP_COCO)
    assert value['preprocessing']['joint_order'] == list(GAITPT_ORDER)
    assert value['preprocessing']['input_shape'] == [2, 30, 18, 2]
    assert value['embedding']['dim'] == 256


def test_semantic_identity_ignores_provenance_but_not_meaning():
    base = contract()
    informational = copy.deepcopy(base)
    informational['provenance'] = {'host': 'nano-lab', 'log': '/tmp/x'}
    informational['runtime_requirements'] = {'torch': '1.10'}
    informational['parity'] = {'status': 'PENDING'}
    assert semantic_id(informational) == semantic_id(base)
    assert semantic_identity(informational, IMPL) == semantic_identity(base, IMPL)
    assert semantic_identity(base, dict(IMPL, backend='e' * 64)) != semantic_identity(base, IMPL)
    assert semantic_id(contract(version='0.10.0')) != semantic_id(base)
    assert semantic_id(contract(asset_sha256=['f' * 64])) != semantic_id(base)
    for section, key, value in (('model', 'checkpoint_sha256', '1' * 64),
                                ('model', 'weights_sha256', '2' * 64),
                                ('preprocessing', 'joint_order', [0] * 18),
                                ('preprocessing', 'quality', 'other'),
                                ('embedding', 'walk_aggregation', 'median'),
                                ('scoring', 'similarity', 'euclidean')):
        changed = copy.deepcopy(base)
        changed[section][key] = value
        assert semantic_id(changed) != semantic_id(base), key


@pytest.mark.parametrize('edit', [
    lambda c: c.update(schema='nano-deployment-contract-v0'),
    lambda c: c.pop('scoring'),
    lambda c: c.update(surprise={}),
    lambda c: c['model'].update(extra=1),
    lambda c: c['model'].pop('family'),
    lambda c: c['model'].update(checkpoint_sha256='not-a-hash'),
    lambda c: c['model'].update(weights_format='torch-pickle'),
    lambda c: c['model'].update(family='Bad Family!'),
    lambda c: c['preprocessing'].update(input_shape=[2, 0, 18, 2]),
    lambda c: c['preprocessing'].update(input_shape='2x30x18x2'),
    lambda c: c['embedding'].update(dim=0),
    lambda c: c['numeric'].update(precision='float16'),
    lambda c: c.update(parity=[]),
])
def test_malformed_contracts_are_refused(edit):
    value = copy.deepcopy(contract())
    edit(value)
    with pytest.raises(ContractError):
        validate_contract(value)
    with pytest.raises(ContractError):
        semantic_id(value)


def test_non_dict_contract_is_refused():
    with pytest.raises(ContractError):
        validate_contract(None)


def test_backend_selection_is_fail_closed_for_unported_families():
    assert backend_for(contract()) is LocalGaitPT
    other = contract()
    other['model']['family'] = 'gpgait'
    with pytest.raises(UnsupportedModel):
        backend_for(other)


def test_input_and_embedding_contract_checks():
    good = np.zeros((2, 30, 18, 2), dtype=np.float32)
    assert check_model_input(good, [2, 30, 18, 2]) is good
    for bad in (good[:1], good.astype(np.float64), np.full_like(good, np.nan),
                np.zeros((2, 30, 17, 2), dtype=np.float32)):
        with pytest.raises(ValueError):
            check_model_input(bad, [2, 30, 18, 2])
    check_embedding(np.zeros(256), 256)
    with pytest.raises(ValueError):
        check_embedding(np.zeros(4864), 256)

"""Numerical parity between reference fixtures from the research implementation and this
runtime. Levels are reported separately because they prove different things: matching
weights or a matching tensor does not imply matching pose, embeddings or accuracy."""
import numpy as np

from .contract import MP_COCO, native_crops, normalize
from .service import prototype_score

SCHEMA = 'gait-parity-fixture-v1'
REPORT_SCHEMA = 'gait-parity-report-v1'
# PROVISIONAL float32 absolute tolerances; the measured differences are in
# docs/RESEARCH_TO_JETSON_MODEL_INTEGRATION.md and must be re-measured on the Nano.
TOLERANCES = dict(model_input=1e-6, walk_embedding=1e-4, reference_prototype=1e-5,
                  similarity=1e-4)
IDENTITY = ('checkpoint_sha256', 'source_revision', 'architecture_sha256')


class ParityError(ValueError):
    pass


def coco17_to_mediapipe33(coco):
    """SYNTHETIC test support: place COCO17 joints at their MediaPipe33 indices and add
    plausible heel/toe points so the Nano quality gate can run. Not a pose conversion."""
    coco = np.asarray(coco, dtype=np.float32)
    mp = np.zeros((len(coco), 33, 2), dtype=np.float32)
    mp[:, list(MP_COCO)] = coco
    for ankle, heel, toe in ((27, 29, 31), (28, 30, 32)):
        mp[:, heel] = mp[:, ankle] + (-4, 6)
        mp[:, toe] = mp[:, ankle] + (12, 8)
    return mp


def check_fixture(fixture, contract):
    if not isinstance(fixture, dict) or fixture.get('schema') != SCHEMA:
        raise ParityError('Unknown parity fixture schema')
    producer = fixture.get('producer') or {}
    if not all(producer.get(k) for k in ('repository', 'commit', 'tool', 'device')):
        raise ParityError('Fixture lacks research-side provenance')
    identity = fixture.get('identity') or {}
    model, pre = contract['model'], contract['preprocessing']
    for key in IDENTITY:
        if identity.get(key) != model[key]:
            raise ParityError('Fixture/deployment model identity differs: ' + key)
    if (identity.get('joint_order') != pre['joint_order']
            or identity.get('input_shape') != pre['input_shape']):
        raise ParityError('Fixture/deployment preprocessing identity differs')
    if not fixture.get('cases'):
        raise ParityError('Fixture has no cases')


def compare(level, actual, expected, atol, expect='match'):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    result = dict(level=level, atol=atol, expect=expect)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return dict(result, status='FAIL', detail='shape %s vs %s or non-finite'
                    % (actual.shape, expected.shape), max_abs_diff=None)
    diff = float(np.abs(actual - expected).max())
    close = diff <= atol
    status = ('PASS' if close else 'FAIL') if expect == 'match' else (
        'FAIL' if close else 'DIFFERS_AS_DOCUMENTED')
    return dict(result, status=status, max_abs_diff=diff)


def run_parity(fixture, contract, encode=None, tolerances=None):
    """`encode(model_input) -> walk embedding` runs this runtime's model; None leaves the
    model-dependent levels NOT_RUN (they are never assumed to pass)."""
    check_fixture(fixture, contract)
    tol = dict(TOLERANCES, **(tolerances or {}))
    results = []
    for case in fixture['cases']:
        expect = case.get('expect', 'match')
        name = case['id']
        vector = None
        coco = np.asarray(case['coco17_xy'], dtype=np.float32)
        crops = native_crops(coco17_to_mediapipe33(coco), np.ones((len(coco), 33), np.float32),
                             np.asarray(case['timestamps']), case['width'], case['height'])
        if 'model_input' in case:
            results.append(dict(compare('model_input', crops, case['model_input'],
                                        tol['model_input']), case=name))
        if encode is not None:
            vector = encode(crops)
        if 'walk_embedding' in case:
            if vector is None:
                results.append(dict(level='walk_embedding', case=name, status='NOT_RUN',
                                    detail='needs model weights/runtime'))
            else:
                results.append(dict(compare('walk_embedding', vector, case['walk_embedding'],
                                            tol['walk_embedding'], expect), case=name))
        if 'reference_embeddings' in case:
            refs = [normalize(v) for v in case['reference_embeddings']]
            prototype = normalize(np.stack(refs).mean(axis=0))
            results.append(dict(compare('reference_prototype', prototype,
                                        case['reference_prototype'], tol['reference_prototype']),
                                case=name))
            basis = vector if vector is not None else case['walk_embedding']
            results.append(dict(compare('similarity', prototype_score(basis, refs),
                                        case['similarity'], tol['similarity']), case=name,
                                basis='hardware-embedding' if vector is not None
                                else 'fixture-embedding'))
    ran = [r for r in results if r['status'] != 'NOT_RUN']
    return dict(schema=REPORT_SCHEMA, producer=fixture['producer'], results=results,
                passed=bool(ran) and all(r['status'] in ('PASS', 'DIFFERS_AS_DOCUMENTED')
                                         for r in ran),
                not_run=sorted(set(r['level'] for r in results if r['status'] == 'NOT_RUN')))

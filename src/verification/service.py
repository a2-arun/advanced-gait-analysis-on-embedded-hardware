"""A query is compared only to the supplied claim's independent references."""
import uuid
import numpy as np

from .contract import InsufficientGait, identifier, normalize
from .policy import decide

MESSAGES = {'OK': 'Gait supports the claimed identity.',
            'CLAIM_NOT_ENROLLED': 'Enroll this claimed identity with the current recipe.',
            'INSUFFICIENT_GAIT': 'Insufficient usable walking gait; recapture.',
            'INVALID_INPUT': 'Invalid input, source lineage, or claimed identity.',
            'PROCESSING_ERROR': 'Local processing failed; inspect the private operator log.',
            'POLICY_NOT_READY': 'No approved calibration policy for this Nano recipe.'}


def response(claim, status, verdict=None):
    message = MESSAGES[status]
    if verdict is False:
        message = 'FALSE: gait does not match the claimed identity.'
    elif verdict is True:
        message = 'TRUE: gait supports the claimed identity.'
    return dict(schema_version='gait-verification-v2', request_id=uuid.uuid4().hex,
                claimed_subject_id=claim, status=status, verdict=verdict, message=message)


def prototype_score(vector, embeddings):
    """Cosine between a walk embedding and the normalized equal-weight reference centroid."""
    prototype = normalize(np.stack(embeddings).mean(axis=0))
    return float(np.dot(normalize(vector), prototype))


class Verifier:
    def __init__(self, store, registry, recipe, encoder_factory, policy=None, logger=None,
                 diagnostic_sink=None):
        self.store, self.registry, self.recipe = store, registry, recipe
        self.encoder_factory, self.policy, self.logger = encoder_factory, policy, logger
        self.diagnostic_sink = diagnostic_sink

    def verify(self, video, claimed_id):
        try:
            identifier(claimed_id)
            query = self.registry.resolve(video)
        except (ValueError, OSError, KeyError, TypeError):
            return response(claimed_id, 'INVALID_INPUT')
        try:
            references = self.store.claim(claimed_id, self.recipe)
            if not references:
                return response(claimed_id, 'CLAIM_NOT_ENROLLED')
            for ref in references:
                self.registry.validate_reference(ref, claimed_id)
                if (ref['take_id'] == query['original_take_id']
                        or ref['source_hash'] == query['sha256']
                        or (query.get('purpose') != 'query' and ref['role'] != query['role'])):
                    return response(claimed_id, 'INVALID_INPUT')
            encoder = self.encoder_factory()
            if encoder.recipe != self.recipe:
                raise ValueError('Encoder recipe changed')
            score = prototype_score(encoder.encode(video), [r['embedding'] for r in references])
            # Recheck sources after processing as well as before it.
            if self.registry.resolve(video) != query:
                raise ValueError('Query changed during processing')
            for ref in references:
                self.registry.validate_reference(ref, claimed_id)
            status, verdict = decide(score, self.recipe, self.policy, reference_count=len(references))
            result = response(claimed_id, status, verdict)
            if self.diagnostic_sink:
                self.diagnostic_sink(dict(schema='nano-private-score-v1',
                    request_id=result['request_id'], recipe=self.recipe,
                    query_take_id=query['original_take_id'], query_sha256=query['sha256'],
                    claimed_subject_id=claimed_id, split_role=query['role'],
                    reference_take_ids=[r['take_id'] for r in references],
                    reference_sha256=[r['source_hash'] for r in references],
                    raw_similarity=score, status=status, verdict=verdict))
            return result
        except InsufficientGait:
            return response(claimed_id, 'INSUFFICIENT_GAIT')
        except Exception:
            if self.logger:
                self.logger.exception('Verification processing failed')
            return response(claimed_id, 'PROCESSING_ERROR')

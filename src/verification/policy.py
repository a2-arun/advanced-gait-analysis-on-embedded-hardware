"""No bundled threshold. Policy and evidence are trusted, operator-controlled files."""
import json
import math
import re
from pathlib import Path

from .contract import fingerprint
from .registry import validate_datetime


def decide(score, recipe, path, reference_count=None):
    blocked = ('POLICY_NOT_READY', None)
    if not path:
        return blocked
    try:
        path = Path(path)
        policy = json.loads(path.read_text())
        threshold = policy['threshold']
        count = policy['reference_count']
        if (policy['schema'] != 'nano-policy-v1' or policy['recipe'] != recipe
                or isinstance(threshold, bool) or not math.isfinite(threshold)
                or not -1 <= threshold <= 1 or not math.isfinite(score)
                or type(count) is not int or count < 1 or count != reference_count):
            return blocked
        evidence = json.loads((path.parent / policy['evidence']).read_text())
        if (fingerprint(evidence) != policy['evidence_sha256']
                or evidence['schema'] != 'nano-calibration-evidence-v1'
                or evidence['recipe'] != recipe or evidence['threshold'] != threshold
                or isinstance(evidence['threshold'], bool)
                or type(evidence['reference_count']) is not int
                or evidence['reference_count'] != count
                or evidence['split_role'] != 'calibration'
                or evidence['validation_status'] != 'APPROVED'
                or not evidence['approved_by']):
            return blocked
        approved = validate_datetime(evidence['approved_at'])
        if approved.utcoffset() is None:
            return blocked
        for key in ('trial_manifest_sha256', 'split_sha256', 'report_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', evidence[key]):
                return blocked
        return 'OK', bool(score >= threshold)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return blocked

"""Structured identity events for downstream automation (RPA, attendance,
access logs, etc). Deliberately decoupled from the ML pipeline: anything
that can produce an IdentificationResult can emit one of these, and nothing
here imports torch/mediapipe. See docs/ARCHITECTURE.md #6 for the intended
event -> RPA boundary.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.model.gait_model import IdentificationResult
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IdentityEvent:
    timestamp: str
    status: str                       # 'IDENTIFIED' or 'UNKNOWN'
    person: Optional[str]
    similarity: float
    threshold: float
    device_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_event(
    result: IdentificationResult, device_id: str = "laptop"
) -> IdentityEvent:
    return IdentityEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="IDENTIFIED" if result.is_identified else "UNKNOWN",
        person=result.identified_name,
        similarity=round(result.similarity, 4),
        threshold=result.threshold,
        device_id=device_id,
    )


def emit_event(event: IdentityEvent, config: Optional[dict] = None) -> Optional[Path]:
    """Log the event and, if configured, write it to outputs/events/ and/or
    POST it to a webhook. Returns the file path written, if any."""
    events_config = (config or load_config())["events"]
    logger.info("Identity event: %s", event.to_dict())

    written_path = None
    if events_config.get("save_to_file", True):
        output_dir = resolve_path(events_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{event.timestamp.replace(':', '-')}.json"
        written_path = output_dir / filename
        written_path.write_text(json.dumps(event.to_dict(), indent=2))

    webhook_url = events_config.get("webhook_url")
    if webhook_url:
        _post_webhook(webhook_url, event)

    return written_path


def _post_webhook(url: str, event: IdentityEvent) -> None:
    try:
        import urllib.request

        body = json.dumps(event.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(request, timeout=5)
    except Exception as exc:
        logger.warning("Failed to deliver event webhook to %s: %s", url, exc)

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

ERROR_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "host_errors.json"

# In-memory cache of recent errors for fast access
_recent_errors: List[Dict[str, Any]] = []

def _load_persisted_errors():
    global _recent_errors
    if ERROR_LOG_PATH.exists():
        try:
            with open(ERROR_LOG_PATH, "r", encoding="utf-8") as f:
                _recent_errors = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load persisted errors: {e}")
            _recent_errors = []

# Initialize on module load
_load_persisted_errors()

def record_error(
    service: str,
    user_message: str,
    technical_details: Optional[str] = None,
    exception: Optional[Exception] = None,
) -> Dict[str, Any]:
    """
    Records a system error into the host diagnostic log.
    Only the host sees these details; user devices receive sanitized messages.
    """
    global _recent_errors
    
    tb_str = ""
    if exception is not None:
        tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    elif technical_details:
        tb_str = technical_details

    error_entry = {
        "id": len(_recent_errors) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "service": service,  # e.g., "Pinecone", "Google Gemini", "PDF Parser", "Network"
        "message": str(user_message),
        "technical_details": tb_str,
        "acknowledged": False,
    }

    _recent_errors.insert(0, error_entry)
    # Keep only the last 100 errors
    if len(_recent_errors) > 100:
        _recent_errors = _recent_errors[:100]

    # Persist to disk
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(_recent_errors, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to persist host error: {e}")

    logger.error(f"[HOST ERROR LOG] Service: {service} | Message: {user_message}")
    return error_entry

def get_recent_errors(limit: int = 20) -> List[Dict[str, Any]]:
    """Returns the most recent system errors for the host dashboard."""
    return _recent_errors[:limit]

def get_unacknowledged_count() -> int:
    """Returns the number of active alerts the host hasn't dismissed."""
    return sum(1 for err in _recent_errors if not err.get("acknowledged", False))

def acknowledge_all_errors():
    """Marks all recorded errors as reviewed by the host."""
    global _recent_errors
    for err in _recent_errors:
        err["acknowledged"] = True
    try:
        if ERROR_LOG_PATH.exists():
            with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(_recent_errors, f, indent=2)
    except Exception:
        pass

def clear_all_errors():
    """Clears the host error log."""
    global _recent_errors
    _recent_errors = []
    try:
        if ERROR_LOG_PATH.exists():
            ERROR_LOG_PATH.unlink(missing_ok=True)
    except Exception:
        pass

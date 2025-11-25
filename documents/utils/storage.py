import os
import tempfile
from typing import Callable, Optional, Tuple

import requests


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        # Ignore cleanup errors – temp files will eventually be removed by OS
        pass


def _get_local_field_path(field_file) -> Optional[str]:
    try:
        path = field_file.path
        return path
    except (NotImplementedError, AttributeError, FileNotFoundError, ValueError):
        return None


def prepare_local_document(document) -> Tuple[str, Callable[[], None]]:
    """
    Ensure a Document.file is accessible from the local filesystem.

    Returns:
        tuple[str, Callable]: (path_to_file, cleanup_callback)
    """
    local_path = _get_local_field_path(document.file)
    if local_path:
        return local_path, lambda: None

    file_url = getattr(document.file, "url", None)
    if not file_url:
        raise RuntimeError("Document file is not accessible via URL.")

    suffix = os.path.splitext(document.file.name or "")[1] or ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp_file, requests.get(
            file_url, stream=True, timeout=30
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)
    except Exception:
        _safe_remove(tmp_path)
        raise

    return tmp_path, lambda: _safe_remove(tmp_path)


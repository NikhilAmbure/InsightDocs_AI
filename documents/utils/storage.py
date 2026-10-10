import os
import glob
import tempfile
import logging
from typing import Callable, List, Optional, Tuple

from django.conf import settings

logger = logging.getLogger(__name__)


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def resolve_local_file_path(document) -> Optional[str]:
    """
    Robust local path resolver for Document files.
    Tries multiple candidate locations in MEDIA_ROOT to locate the file on disk.
    """
    # 1. Direct FieldFile.path
    try:
        if hasattr(document, "file") and document.file:
            path = document.file.path
            if path and os.path.exists(path):
                return path
    except (NotImplementedError, AttributeError, ValueError, OSError):
        pass

    # Candidate filenames and subpaths to check
    media_root = str(settings.MEDIA_ROOT)
    file_name = getattr(document.file, "name", "") if hasattr(document, "file") and document.file else ""
    original_name = getattr(document, "original_name", "") or ""
    title = getattr(document, "title", "") or ""

    candidates = []

    if file_name:
        candidates.append(os.path.join(media_root, file_name))
        candidates.append(os.path.join(str(settings.BASE_DIR), file_name))
        candidates.append(os.path.join(media_root, "documents", os.path.basename(file_name)))
        # Strip potential Cloudinary random suffixes (e.g. filename_lf4bux -> filename.pdf)
        base_without_suffix = file_name.rsplit("_", 1)[0]
        ext = getattr(document, "extension", "")
        if ext:
            candidates.append(os.path.join(media_root, "documents", f"{os.path.basename(base_without_suffix)}.{ext}"))

    if original_name:
        candidates.append(os.path.join(media_root, "documents", original_name))
        candidates.append(os.path.join(media_root, original_name))

    if title:
        candidates.append(os.path.join(media_root, "documents", title))

    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.path.isfile(candidate):
            return candidate

    # Search directory for prefix match if exact name not matched
    docs_dir = os.path.join(media_root, "documents")
    if os.path.exists(docs_dir):
        stem = os.path.splitext(os.path.basename(file_name))[0] if file_name else ""
        if stem:
            # Match files starting with the same stem
            matches = glob.glob(os.path.join(docs_dir, f"{stem}*"))
            for m in matches:
                if os.path.isfile(m):
                    return m

        # Match by original_name stem
        orig_stem = os.path.splitext(os.path.basename(original_name))[0] if original_name else ""
        if orig_stem:
            matches = glob.glob(os.path.join(docs_dir, f"*{orig_stem}*"))
            for m in matches:
                if os.path.isfile(m):
                    return m

    return None


def prepare_local_document(document) -> Tuple[str, Callable[[], None]]:
    """
    Ensure a Document.file is accessible on the local filesystem for processing
    (PyMuPDF, docx, pptx, OCR, etc.).
    
    1. Returns direct path if local on disk.
    2. Writes from DB content or storage open to a temporary file if needed.
    
    Returns:
        tuple[str, Callable]: (path_to_file, cleanup_callback)
    """
    # 1. Check if file exists locally on disk
    local_path = resolve_local_file_path(document)
    if local_path:
        return local_path, lambda: None

    # 2. Check if we have content stored in DB (BinaryField)
    if hasattr(document, "file_content") and document.file_content:
        suffix = os.path.splitext(getattr(document.file, "name", "") or "")[1]
        if not suffix and getattr(document, "extension", ""):
            suffix = f".{document.extension}"
        suffix = suffix or ".tmp"

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(document.file_content)
            return tmp_path, lambda: _safe_remove(tmp_path)
        except Exception:
            _safe_remove(tmp_path)
            raise

    # 3. Read via Django storage backend .open()
    if hasattr(document, "file") and document.file:
        suffix = os.path.splitext(getattr(document.file, "name", "") or "")[1]
        if not suffix and getattr(document, "extension", ""):
            suffix = f".{document.extension}"
        suffix = suffix or ".tmp"

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with document.file.open("rb") as f:
                with os.fdopen(fd, "wb") as tmp_file:
                    while chunk := f.read(1024 * 1024):
                        tmp_file.write(chunk)
            return tmp_path, lambda: _safe_remove(tmp_path)
        except Exception as e:
            _safe_remove(tmp_path)
            logger.error(f"Failed to prepare local copy for document {document.id}: {e}")
            raise

    raise FileNotFoundError(f"Document {document.id} file could not be found locally on disk.")


def get_candidate_urls(document) -> List[str]:
    """
    Returns candidate URLs for serving the document locally.
    """
    url = getattr(document.file, "url", "") if hasattr(document, "file") and document.file else ""
    return [url] if url else []
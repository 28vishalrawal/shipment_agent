"""Upload validation: size, type, and a malware-scan integration point."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings

ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls"}


async def scan_for_malware(data: bytes) -> bool:
    """Integration point for ClamAV / cloud AV. Returns True if clean.

    Assumption: no scanner is wired in the starter; we return True but keep the
    seam so production can drop in a real client without touching callers.
    """
    return True


async def read_upload(file: UploadFile) -> pd.DataFrame:
    s = get_settings()
    filename = file.filename or ""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported type {suffix}")

    data = await file.read()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file too large")

    if not await scan_for_malware(data):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "failed malware scan")

    try:
        if suffix == ".csv":
            # Try decoding as UTF-8 first, then fallback to ISO-8859-1 and hp-roman8
            for encoding in ("utf-8", "ISO-8859-1", "hp-roman8"):
                try:
                    return pd.read_csv(io.BytesIO(data), encoding=encoding)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # For other errors (e.g., pd parsing error), raise as usual
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"unreadable file: {e}") from e
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unreadable file: Could not decode CSV with utf-8, ISO-8859-1, or hp-roman8 encodings.")
        return pd.read_excel(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"unreadable file: {exc}") from exc

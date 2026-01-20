from __future__ import annotations

import os
from typing import Optional


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))

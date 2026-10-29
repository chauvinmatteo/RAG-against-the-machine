import os
from typing import List


def extract_files(root: str) -> List[str]:
    """Recursively collect the indexable files under a directory.

    Args:
        root: Directory to scan (e.g. "data/raw").

    Returns:
        Sorted list of file paths, relative to the project root.
    """
    files: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith((".py", ".md")):
                files.append(os.path.join(dirpath, name))
    return sorted(files)

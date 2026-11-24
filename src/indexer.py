import os
from typing import List, Optional, Tuple

from pydantic import BaseModel


class Chunk(BaseModel):
    """A slice of a corpus file.

    text == file_content[first_character_index:last_character_index]
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    headers: str = ""


class ChunkIndex(BaseModel):
    max_chunk_size: int
    chunks: List[Chunk]


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


def _parse_header(line: str) -> Optional[Tuple[int, str]]:
    """Return (level, title) if the line is a markdown header, else None."""
    stripped = line.rstrip("\n")
    level = len(stripped) - len(stripped.lstrip("#"))
    if 1 <= level <= 6 and stripped[level:level + 1] == " ":
        return level, stripped[level:].strip()
    return None


def _is_fence(line: str) -> bool:
    """Return True if the line opens or closes a code block."""
    return line.lstrip().startswith(("```", "~~~"))


def _split_sections(text: str) -> List[Tuple[int, int, str]]:
    """Split markdown on headers, ignoring '#' lines inside code blocks.

    Returns:
        List of (start, end, header_path), end excluded.
    """
    sections: List[Tuple[int, int, str]] = []
    stack: List[Tuple[int, str]] = []
    in_code = False
    sec_start, sec_path = 0, ""
    pos = 0
    for line in text.splitlines(keepends=True):
        if _is_fence(line):
            in_code = not in_code
        header = None if in_code else _parse_header(line)
        if header:
            if pos > sec_start:
                sections.append((sec_start, pos, sec_path))
            level, title = header
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sec_start = pos
            sec_path = " > ".join(t for _, t in stack)
        pos += len(line)
    if pos > sec_start:
        sections.append((sec_start, pos, sec_path))
    return sections


def _split_big(text: str, start: int, end: int,
               max_size: int) -> List[Tuple[int, int]]:
    """Split [start, end) into ranges of at most max_size characters.

    Cuts on a blank line if possible, then on a newline, then anywhere.
    """
    ranges: List[Tuple[int, int]] = []
    while end - start > max_size:
        cut = start + max_size
        for sep in ("\n\n", "\n"):
            idx = text.rfind(sep, start, start + max_size)
            if idx > start:
                cut = idx + len(sep)
                break
        ranges.append((start, cut))
        start = cut
    ranges.append((start, end))
    return ranges


def _build_chunks(file_path: str, text: str,
                  sections: List[Tuple[int, int, str]],
                  max_size: int, min_size: int) -> List[Chunk]:
    """Turn (start, end, headers) sections into chunks <= max_size.

    Big sections are split with _split_big, and pieces smaller than
    min_size are merged into the previous chunk when it still fits.
    """
    chunks: List[Chunk] = []
    for s, e, path in sections:
        for cs, ce in _split_big(text, s, e, max_size):
            if (chunks and ce - cs < min_size
                    and ce - chunks[-1].first_character_index <= max_size):
                prev = chunks[-1]
                prev.last_character_index = ce
                prev.text = text[prev.first_character_index:ce]
                continue
            if text[cs:ce].strip():
                chunks.append(Chunk(file_path=file_path,
                                    first_character_index=cs,
                                    last_character_index=ce,
                                    text=text[cs:ce], headers=path))
    return chunks


def chunk_markdown(file_path: str, text: str, max_size: int = 2000,
                   min_size: int = 200) -> List[Chunk]:
    """Split a markdown file into chunks of at most max_size characters.

    Args:
        file_path: Path of the file, stored as is in each chunk.
        text: Full content of the file.
        max_size: Maximum number of characters per chunk.
        min_size: Sections smaller than this are merged into the previous
            chunk when the result still fits in max_size.

    Returns:
        The chunks, in file order.
    """
    return _build_chunks(file_path, text, _split_sections(text),
                         max_size, min_size)


def _is_block_start(line: str, indent: str) -> bool:
    """Return True if line starts a def / class / decorator at this indent."""
    if not line.startswith(indent):
        return False
    rest = line[len(indent):]
    return rest.startswith(("def ", "async def ", "class ", "@"))


def _python_sections(text: str, start: int, end: int, indent: str,
                     prefix: str,
                     max_size: int) -> List[Tuple[int, int, str]]:
    """Split [start, end) on the def / class found at this indent.

    Lines inside triple-quoted strings are ignored. A class bigger than
    max_size is split again on its methods (one indent level deeper).

    Returns:
        List of (start, end, header_path), end excluded.
    """
    sections: List[Tuple[int, int, str]] = []
    cur_start, cur_name = start, prefix
    in_string = False
    after_decorator = False
    pos = start
    for line in text[start:end].splitlines(keepends=True):
        if not in_string and _is_block_start(line, indent):
            rest = line[len(indent):]
            if not after_decorator:
                if pos > cur_start:
                    sections.append((cur_start, pos, cur_name))
                cur_start = pos
            after_decorator = rest.startswith("@")
            if not after_decorator:
                name = rest.split("(")[0].split(":")[0].strip()
                cur_name = f"{prefix} > {name}" if prefix else name
        if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
            in_string = not in_string
        pos += len(line)
    if end > cur_start:
        sections.append((cur_start, end, cur_name))

    result: List[Tuple[int, int, str]] = []
    for s, e, name in sections:
        is_class = name.split(" > ")[-1].startswith("class ")
        # name != prefix: only new blocks, else infinite recursion
        if name != prefix and is_class and e - s > max_size:
            result += _python_sections(text, s, e, indent + "    ",
                                       name, max_size)
        else:
            result.append((s, e, name))
    return result


def chunk_python(file_path: str, text: str, max_size: int = 2000,
                 min_size: int = 200) -> List[Chunk]:
    """Split a Python file into chunks of at most max_size characters.

    Chunks follow top-level functions and classes (and methods of big
    classes), found line by line from their indentation.

    Args:
        file_path: Path of the file, stored as is in each chunk.
        text: Full content of the file.
        max_size: Maximum number of characters per chunk.
        min_size: Sections smaller than this are merged into the previous
            chunk when the result still fits in max_size.

    Returns:
        The chunks, in file order.
    """
    sections = _python_sections(text, 0, len(text), "", "", max_size)
    return _build_chunks(file_path, text, sections, max_size, min_size)

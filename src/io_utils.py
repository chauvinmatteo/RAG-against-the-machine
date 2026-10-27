import json

from pydantic import ValidationError, BaseModel
from src.models import RagDataset, StudentSearchResults


class DataLoadError(Exception):
    """Raised when a data file cannot be read or does not match its schema."""


class JSONLoadError(DataLoadError):
    """Raised when a data file does not contain valid JSON."""


def _read_text(path: str) -> str:
    """Read a UTF-8 text file and return its content.

    Args:
        path: Path to the file.

    Returns:
        The file content as a string.

    Raises:
        DataLoadError: If the file cannot be opened or decoded.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        raise DataLoadError(f"Cannot read '{path}': {e}") from e


def load_dataset(path: str) -> RagDataset:
    """Load and validate a RAG dataset from a JSON file.

    Args:
        path: Path to the dataset JSON file.

    Returns:
        The validated dataset.

    Raises:
        JSONLoadError: If the file does not contain valid JSON.
        DataLoadError: If the file cannot be read or is not a valid dataset.
    """
    text = _read_text(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise JSONLoadError(f"Invalid json '{path}': {e}") from e
    try:
        return RagDataset.model_validate(data)
    except ValidationError as e:
        raise DataLoadError(f"Invalid dataset '{path}': "
                            f"{e.errors()[0]['loc']} "
                            f"{e.errors()[0]['msg']}") from e


def load_search_results(path: str) -> StudentSearchResults:
    """Load and validate search results from a JSON file.

    Args:
        path: Path to the search results JSON file.

    Returns:
        The validated search results.

    Raises:
        JSONLoadError: If the file does not contain valid JSON.
        DataLoadError: If the file cannot be read or does not contain
            valid search results.
    """
    text = _read_text(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise JSONLoadError(f"Invalid json '{path}': {e}") from e
    try:
        return StudentSearchResults.model_validate(data)
    except ValidationError as e:
        raise DataLoadError(f"Invalid search results '{path}': "
                            f"{e.errors()[0]['loc']} "
                            f"{e.errors()[0]['msg']}") from e


def save_model(model: BaseModel, directory: str, filename: str) -> None:
    pass

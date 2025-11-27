import json
from typing import Any, Dict, List
from pathlib import Path


def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    Load a JSON file from the given file path.

    Args:
        filepath (str): Path to the JSON file to load.

    Returns:
        Dict[str, Any]: Dictionary containing the parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        PermissionError: If there are insufficient permissions to read the file.
    """

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in file {filepath}: {e.msg}", e.doc, e.pos
        )
    except PermissionError:
        raise PermissionError(f"Permission denied when reading file: {filepath}")


def get_json_files(directory_path: str) -> List[str]:
    """
    Get a list of all JSON files in the specified directory.

    Args:
        directory_path (str): Path to the directory to search for JSON files.

    Returns:
        List[str]: List of absolute paths to all JSON files found in the directory.
            Returns an empty list if no JSON files are found.

    Raises:
        FileNotFoundError: If the directory does not exist.
        PermissionError: If there are insufficient permissions to access the directory.
    """

    path = Path(directory_path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")

    try:
        # Use glob for search of .json files
        json_files = [str(file.resolve()) for file in path.glob("*.json")]
        return sorted(json_files)
    except PermissionError:
        raise PermissionError(
            f"Permission denied when accessing directory: {directory_path}"
        )


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data (Dict[str, Any]): Dictionary to save.
        filepath (str): Path where JSON file will be saved.
    """

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

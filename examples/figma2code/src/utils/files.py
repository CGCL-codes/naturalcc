import json
from pathlib import Path

# ==================== JSON Utility Functions ====================
def load_json(file_path: str | Path) -> dict | None:
    """
    Load a JSON file from the specified path.
    
    Args:
        file_path (str | Path): JSON file path
        
    Returns:
        dict | None: JSON data, or None if the file does not exist or parsing fails
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def save_json(data: dict, file_path: str | Path) -> bool:
    """
    Save data as a JSON file.
    
    Args:
        data (dict): Data to be saved
        file_path (str | Path): Save path
        
    Returns:
        bool: Whether the save was successful
    """
    try:
        # Ensure the directory exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False
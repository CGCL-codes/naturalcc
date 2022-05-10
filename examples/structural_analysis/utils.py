from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent


def get_data_path() -> Path:
    """Returns data root folder."""
    return get_project_root() / Path('data')


def get_models_path() -> Path:
    """Returns models root folder."""
    return get_project_root() / Path('models')


def get_reports_path() -> Path:
    """Returns models root folder."""
    return get_project_root() / Path('reports')


def get_cache_path() -> Path:
    """Returns data root folder."""
    return get_data_path() / Path('cache')


def dataset():
    return None
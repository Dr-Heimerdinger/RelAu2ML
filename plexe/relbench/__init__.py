import importlib as _importlib

from . import base, datasets, tasks
from .base import (
    Database,
    Dataset,
    Table,
    BaseTask,
    TaskType,
    EntityTask,
    RecommendationTask,
    AutoCompleteTask,
)

__version__ = "1.1.0"

__all__ = [
    "base",
    "datasets",
    "modeling",
    "tasks",
    "Database",
    "Dataset",
    "Table",
    "BaseTask",
    "TaskType",
    "EntityTask",
    "RecommendationTask",
    "AutoCompleteTask",
]


def __getattr__(name: str):
    """Lazy-import 'modeling' so that dataset.py / task.py (which only need
    plexe.relbench.base) don't pull in torch_frame and its heavy deps."""
    if name == "modeling":
        return _importlib.import_module(".modeling", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

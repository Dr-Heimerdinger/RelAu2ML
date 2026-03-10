from typing import Dict
from langchain_core.tools import tool as langchain_tool

@langchain_tool
def save_artifact(
    content: str,
    filename: str,
    working_dir: str
) -> Dict[str, str]:
    """
    Save an artifact file to the working directory.

    Args:
        content: File content to write
        filename: Name of the file (e.g. "dataset.py"). If a full path
                  containing working_dir is passed, only the relative part is used.
        working_dir: Working directory for the session

    Returns:
        Save status and file path
    """
    import os

    # Guard against path doubling: if the agent passes the full path as
    # filename (e.g. "workdir/session-.../dataset.py"), strip the
    # working_dir prefix so we don't create
    # "workdir/.../workdir/.../dataset.py".
    if working_dir and filename.startswith(working_dir):
        filename = os.path.relpath(filename, working_dir)

    os.makedirs(working_dir, exist_ok=True)
    file_path = os.path.join(working_dir, filename)

    # Ensure parent directories exist (in case filename has subdirs)
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write(content)

    return {
        "status": "saved",
        "file_path": file_path
    }

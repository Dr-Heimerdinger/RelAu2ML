import os
import re
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Kaggle")

@mcp.tool()
def search_gnn_competitions_for_benchmarks(
    task_type: str = "graph",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search Kaggle competitions for GNN benchmarks and winning solutions.
    
    Useful for finding proven hyperparameters from competition winners.
    
    Args:
        task_type: Type of task (e.g., "graph", "node classification", "link prediction")
        limit: Maximum number of competitions to retrieve
    
    Returns:
        Dict with competition info and potential hyperparameter insights
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    # Search for competitions
    query = f"{task_type} neural network"
    competitions = api.competitions_list(search=query)
    
    results = []
    for i, comp in enumerate(competitions):
        if i >= limit:
            break
        results.append({
            "ref": comp.ref,
            "title": comp.title,
            "description": comp.description[:200] if comp.description else "",
            "reward": str(comp.reward) if hasattr(comp, 'reward') else "N/A",
            "teamCount": comp.teamCount if hasattr(comp, 'teamCount') else 0,
            "userHasEntered": comp.userHasEntered if hasattr(comp, 'userHasEntered') else False
        })
    
    return {
        "competitions_found": len(results),
        "competitions": results,
        "source": "Kaggle Competitions",
        "note": "Check competition notebooks for winning hyperparameters"
    }

@mcp.tool()
def search_gnn_notebooks_for_hyperparameters(
    task_type: str = "graph neural network",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search Kaggle notebooks for GNN implementations and extract hyperparameters.
    
    Args:
        task_type: Type of task or model (e.g., "GNN", "Graph Neural Network")
        limit: Maximum number of notebooks to retrieve
    
    Returns:
        Dict with notebook info and hyperparameter insights
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    # Search for notebooks/kernels
    kernels = api.kernels_list(search=task_type, sort_by="voteCount")
    
    results = []
    for i, kernel in enumerate(kernels):
        if i >= limit:
            break
        results.append({
            "ref": kernel.ref,
            "title": kernel.title,
            "author": kernel.author,
            "voteCount": kernel.voteCount if hasattr(kernel, 'voteCount') else 0,
            "language": kernel.language if hasattr(kernel, 'language') else "unknown",
            "url": f"https://www.kaggle.com/{kernel.ref}"
        })
    
    return {
        "notebooks_found": len(results),
        "notebooks": results,
        "source": "Kaggle Notebooks",
        "confidence": "high" if results else "low",
        "note": "Top notebooks by votes often contain well-tuned hyperparameters"
    }

@mcp.tool()
def search_kaggle_datasets(query: str, limit: int = 5):
    """
    Search for datasets on Kaggle.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    datasets = api.dataset_list(search=query)
    results = []
    for i, ds in enumerate(datasets):
        if i >= limit:
            break
        results.append({
            "ref": ds.ref,
            "title": ds.title,
            "size": ds.size,
            "lastUpdated": str(ds.lastUpdated),
            "downloadCount": ds.downloadCount,
            "voteCount": ds.voteCount
        })
    return results

@mcp.tool()
def download_kaggle_dataset(dataset_ref: str, path: str = "data"):
    """
    Download a Kaggle dataset.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    os.makedirs(path, exist_ok=True)
    api.dataset_download_files(dataset_ref, path=path, unzip=True)
    return f"Dataset {dataset_ref} downloaded to {path}"

if __name__ == "__main__":
    mcp.run()

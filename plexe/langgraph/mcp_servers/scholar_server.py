import sys
import re
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP
from scholarly import scholarly

# Initialize FastMCP server
mcp = FastMCP("Google Scholar")

@mcp.tool()
def search_gnn_papers_for_hyperparameters(
    task_type: str,
    model_type: str = "Graph Neural Network",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search Google Scholar for GNN papers and extract hyperparameters.
    
    Optimized for finding optimal hyperparameters for GNN training.
    
    Args:
        task_type: Type of task (e.g., "node classification", "link prediction", "graph classification")
        model_type: Model architecture (default: "Graph Neural Network")
        limit: Maximum number of papers to analyze
    
    Returns:
        Dict with papers and extracted hyperparameters
    """
    query = f"{model_type} {task_type} hyperparameters learning rate"
    search_query = scholarly.search_pubs(query)
    
    papers = []
    hyperparams_found = []
    
    for i, pub in enumerate(search_query):
        if i >= limit:
            break
        
        bib = pub.get('bib', {})
        abstract = bib.get('abstract', '')
        
        paper_info = {
            "title": bib.get('title'),
            "authors": bib.get('author'),
            "year": bib.get('pub_year'),
            "venue": bib.get('venue'),
            "citations": pub.get('num_citations', 0),
            "url": pub.get('pub_url'),
            "abstract": abstract[:500] if abstract else ""  # Truncate for display
        }
        papers.append(paper_info)
        
        # Extract hyperparameters from abstract
        if abstract:
            extracted = _extract_hyperparameters_from_text(abstract)
            if extracted:
                hyperparams_found.append({
                    "paper": bib.get('title', 'Unknown'),
                    "hyperparameters": extracted
                })
    
    # Aggregate findings
    aggregated = _aggregate_hyperparameters(hyperparams_found)
    
    return {
        "papers_analyzed": len(papers),
        "papers_with_hyperparams": len(hyperparams_found),
        "papers": papers,
        "extracted_hyperparameters": hyperparams_found,
        "aggregated_recommendations": aggregated,
        "confidence": "high" if len(hyperparams_found) >= 3 else "medium",
        "source": "Google Scholar"
    }

@mcp.tool()
def search_scholar(query: str, limit: int = 5):
    """
    General search for academic papers on Google Scholar.
    
    Args:
        query: Search query
        limit: Maximum number of results
    """
    search_query = scholarly.search_pubs(query)
    results = []
    for i, pub in enumerate(search_query):
        if i >= limit:
            break
        bib = pub.get('bib', {})
        results.append({
            "title": bib.get('title'),
            "author": bib.get('author'),
            "pub_year": bib.get('pub_year'),
            "venue": bib.get('venue'),
            "abstract": bib.get('abstract'),
            "url": pub.get('pub_url'),
            "num_citations": pub.get('num_citations')
        })
    return results

@mcp.tool()
def get_author_info(name: str):
    """
    Get information about an academic author on Google Scholar.
    """
    search_query = scholarly.search_author(name)
    author = next(search_query, None)
    if author:
        author = scholarly.fill(author)
        return {
            "name": author.get('name'),
            "affiliation": author.get('affiliation'),
            "interests": author.get('interests'),
            "citedby": author.get('citedby'),
            "hindex": author.get('hindex'),
            "publications_count": len(author.get('publications', []))
        }
    return "Author not found"

def _extract_hyperparameters_from_text(text: str) -> Dict[str, Any]:
    """Extract hyperparameter values from text using regex patterns."""
    hyperparams = {}
    text_lower = text.lower()
    
    # Learning rate
    lr_patterns = [
        r'learning rate[:\s]+([0-9.e-]+)',
        r'lr[:\s=]+([0-9.e-]+)',
    ]
    for pattern in lr_patterns:
        match = re.search(pattern, text_lower)
        if match:
            hyperparams['learning_rate'] = float(match.group(1))
            break
    
    # Batch size
    batch_patterns = [
        r'batch size[:\s]+([0-9]+)',
        r'batch[:\s=]+([0-9]+)',
    ]
    for pattern in batch_patterns:
        match = re.search(pattern, text_lower)
        if match:
            hyperparams['batch_size'] = int(match.group(1))
            break
    
    # Hidden dimensions/channels
    hidden_patterns = [
        r'hidden[_ ](?:dimension|channel|unit)s?[:\s]+([0-9]+)',
        r'embedding[_ ]size[:\s]+([0-9]+)',
    ]
    for pattern in hidden_patterns:
        match = re.search(pattern, text_lower)
        if match:
            hyperparams['hidden_channels'] = int(match.group(1))
            break
    
    # Number of layers
    layer_patterns = [
        r'([0-9]+)[- ]layer',
        r'num[_ ]layers?[:\s]+([0-9]+)',
    ]
    for pattern in layer_patterns:
        match = re.search(pattern, text_lower)
        if match:
            hyperparams['num_layers'] = int(match.group(1))
            break
    
    # Epochs
    epoch_patterns = [
        r'([0-9]+) epochs?',
        r'epochs?[:\s]+([0-9]+)',
    ]
    for pattern in epoch_patterns:
        match = re.search(pattern, text_lower)
        if match:
            hyperparams['epochs'] = int(match.group(1))
            break
    
    return hyperparams

def _aggregate_hyperparameters(hyperparams_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate hyperparameters from multiple papers using median."""
    if not hyperparams_list:
        return {}
    
    aggregated = {}
    all_params = set()
    
    for item in hyperparams_list:
        hp = item.get('hyperparameters', {})
        all_params.update(hp.keys())
    
    for param in all_params:
        values = []
        for item in hyperparams_list:
            hp = item.get('hyperparameters', {})
            if param in hp:
                values.append(hp[param])
        
        if values:
            # Take median for numeric values
            sorted_vals = sorted(values)
            aggregated[param] = sorted_vals[len(sorted_vals) // 2]
    
    return aggregated

if __name__ == "__main__":
    mcp.run()

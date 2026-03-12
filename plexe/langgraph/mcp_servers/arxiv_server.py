import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("arXiv")

BASE_URL = "http://export.arxiv.org/api/query"

@mcp.tool()
def search_arxiv_papers(
    query: str,
    max_results: int = 10,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers.
    """
    sort_by = sort_by or "relevance"
    sort_order = sort_order or "descending"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': sort_order
    }
    
    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return _parse_arxiv_response(response.text)
    except Exception as e:
        return [{"error": str(e)}]

def _parse_arxiv_response(xml_text: str) -> List[Dict[str, Any]]:
    """Parse arXiv XML response into list of paper dicts."""
    papers = []
    
    try:
        root = ET.fromstring(xml_text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            paper = {
                'id': entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else '',
                'title': entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else '',
                'summary': entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else '',
                'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else '',
                'updated': entry.find('atom:updated', ns).text if entry.find('atom:updated', ns) is not None else '',
                'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                'categories': [cat.get('term') for cat in entry.findall('atom:category', ns)]
            }
            papers.append(paper)
    except Exception as e:
        papers.append({"error": f"Parse error: {str(e)}"})
    
    return papers

if __name__ == "__main__":
    mcp.run()

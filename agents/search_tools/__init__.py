"""
Search tools for agents.

Provides abstract search interface and implementations for different backends.
"""

from .base import BaseSearchTool, SearchResult, Article
from .chunking import chunk_text, chunk_article
from .google_search import GoogleNewsSearchTool, create_search_tool

__all__ = [
    'BaseSearchTool',
    'SearchResult',
    'Article',
    'chunk_text',
    'chunk_article',
    'GoogleNewsSearchTool',
    'create_search_tool',
]

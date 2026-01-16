"""
Triple-Hybrid-RAG Graph Module

Native PuppyGraph integration using Bolt/Cypher protocol.
"""

from triple_hybrid_rag.graph.puppygraph import PuppyGraphClient
from triple_hybrid_rag.graph.schema import GraphSchema

__all__ = [
    "PuppyGraphClient",
    "GraphSchema",
]

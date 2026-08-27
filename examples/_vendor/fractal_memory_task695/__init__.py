from .macro_graph import MacroGraph, ConceptNode, Edge
from .micro_list import MicroListStore, MicroItem
from .cluster_splitter import ClusterSplitter
from .consolidation_engine import ConsolidationEngine
from .fractal_ltm import FractalLTM
from .vector_adapters import (
    BaseVectorAdapter,
    InMemoryVectorAdapter,
    ChromaVectorAdapter,
    PineconeVectorAdapter,
    PgVectorAdapter
)

__all__ = [
    "MacroGraph",
    "ConceptNode",
    "Edge",
    "MicroListStore",
    "MicroItem",
    "ClusterSplitter",
    "ConsolidationEngine",
    "FractalLTM",
    "BaseVectorAdapter",
    "InMemoryVectorAdapter",
    "ChromaVectorAdapter",
    "PineconeVectorAdapter",
    "PgVectorAdapter"
]

# v2 needs chromadb / numpy / scikit-learn; keep it optional so the v1 modules and
# their tests still import on an interpreter that only has the stdlib.
try:
    from .v2 import FractalLTMv2  # noqa: F401
    __all__.append("FractalLTMv2")
except ImportError:  # pragma: no cover
    pass

from .base import MemoryProvider
from .hippocampus import HippocampusMemoryProvider
from .none import NoMemoryProvider

REGISTRY: dict[str, type[MemoryProvider]] = {
    "vanilla": NoMemoryProvider,
    "hippocampus": HippocampusMemoryProvider,
}
# legacy alias (docs/scripts used this); canonical name above
REGISTRY["none"] = NoMemoryProvider

# Providers with optional third-party dependencies: registered only when importable, so a
# minimal install can still run stdlib-only providers (none, hippocampus).
_OPTIONAL: dict[str, tuple[str, str]] = {
    # registry name -> (module, class name)
    "hindsight-coding": ("hscoding", "HsCodingProvider"),
    "bm25": ("bm25", "BM25MemoryProvider"),
    "cognee": ("cognee", "CogneeMemoryProvider"),
    "hindsight": ("hindsight", "HindsightMemoryProvider"),
    "hindsight-cloud": ("hindsight", "HindsightCloudMemoryProvider"),
    "hindsight-http": ("hindsight", "HindsightHTTPMemoryProvider"),
    "mastra": ("mastra", "MastraMemoryProvider"),
    "mastra-om": ("mastra_om", "MastraOMMemoryProvider"),
    "mem0": ("mem0", "Mem0MemoryProvider"),
    "mem0-cloud": ("mem0_cloud", "Mem0CloudMemoryProvider"),
    "ogham": ("ogham", "OghamMemoryProvider"),
    "qdrant": ("hybrid_search", "HybridSearchMemoryProvider"),
    "supermemory": ("supermemory", "SupermemoryMemoryProvider"),
}

UNAVAILABLE: dict[str, str] = {}
for _name, (_module, _cls) in _OPTIONAL.items():
    try:
        REGISTRY[_name] = getattr(__import__(f"{__name__}.{_module}", fromlist=[_cls]), _cls)
    except ImportError as exc:
        UNAVAILABLE[_name] = str(exc)
if "hindsight-coding" in REGISTRY:  # legacy alias
    REGISTRY["hscoding"] = REGISTRY["hindsight-coding"]


def get_memory_provider(name: str) -> MemoryProvider:
    if name not in REGISTRY:
        hint = f" (installed but dependency missing: {UNAVAILABLE[name]})" if name in UNAVAILABLE else ""
        raise ValueError(f"Unknown memory provider: '{name}'{hint}. Available: {list(REGISTRY)}")
    return REGISTRY[name]()

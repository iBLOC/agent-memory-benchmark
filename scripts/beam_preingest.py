#!/usr/bin/env python
"""Pre-ingest BEAM/100k into the :8083 server using the provider's own logic,
so --skip-ingestion harness runs reuse the store + sidecar verbatim.

Run:  PYTHONPATH=src python scripts/beam_preingest.py
Requires the :8083 server up (HIPPOCAMPUS_URL override honored).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HIPPOCAMPUS_URL", "http://127.0.0.1:8083")

from memory_bench.dataset import get_dataset  # noqa: E402
from memory_bench.memory.hippocampus import HippocampusMemoryProvider  # noqa: E402

STORE = Path("outputs/beam/hippo-beam-mimo/_store/100k/all")


def main() -> None:
    ds = get_dataset("beam")
    docs = ds.load_documents("100k")
    print(f"BEAM 100k documents: {len(docs)}", flush=True)

    provider = HippocampusMemoryProvider()
    provider.prepare(STORE, unit_ids={d.user_id for d in docs if d.user_id}, reset=True)

    t0 = time.time()
    batch = 2000
    for i in range(0, len(docs), batch):
        provider.ingest(docs[i : i + batch])
        done = min(i + batch, len(docs))
        rate = done / max(time.time() - t0, 1)
        print(f"  {done}/{len(docs)}  ({rate:.0f} docs/s, eta {((len(docs)-done)/max(rate,1))/60:.0f} min)", flush=True)

    print(f"pre-ingest complete in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

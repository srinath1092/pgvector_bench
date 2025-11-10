import re
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd

# --- Data Classes ---

@dataclass
class BaseEntry:
    raw: str
    line_no: int
    timestamp: Optional[str] = None
    duration: Optional[str] = None  # formatted in ms

@dataclass
class ExactSearch(BaseEntry):
    dist_func: str = ""
    topk: int = 0
    query_count: int = 0

@dataclass
class HNSWBuild(BaseEntry):
    dist_func: str = ""
    max_cons: int = 0
    ef_construction: int = 0

@dataclass
class HNSWQueryEnhanced(BaseEntry):
    ef_search: int = 0
    dist_func: str = ""
    topk: int = 0
    query_count: int = 0
    max_cons: int = 0
    ef_construction: int = 0

@dataclass
class HNSWRecall(BaseEntry):
    ef_search: int = 0
    max_cons: int = 0
    ef_construction: int = 0
    query_count: int = 0

@dataclass
class IVFBuild(BaseEntry):
    dist_func: str = ""
    ivf_list_count: int = 0

@dataclass
class IVFQueryEnhanced(BaseEntry):
    n_probes: int = 0
    dist_func: str = ""
    topk: int = 0
    query_count: int = 0
    ivf_list_count: int = 0

@dataclass
class IVFRecall(BaseEntry):
    ivf_list_count: int = 0
    n_probes: int = 0
    query_count: int = 0


# --- Helpers for extracting time info ---
ts_patterns = [
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)",
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]",
    r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
]

dur_patterns = [
    r"(\d+(?:\.\d+)?)\s*ms\b",
    r"(\d+(?:\.\d+)?)\s*milliseconds?\b",
    r"(\d+(?:\.\d+)?)\s*s\b",
    r"took\s*(\d+(?:\.\d+)?)\s*s\b",
    r"time[:=]\s*(\d+(?:\.\d+)?)\s*s\b",
    r"duration[:=]\s*(\d+(?:\.\d+)?)\s*s\b",
]

def extract_time_info(line: str):
    ts = None
    dur_s = None
    for p in ts_patterns:
        if m := re.search(p, line):
            ts = m.group(1)
            break
    for p in dur_patterns:
        if m := re.search(p, line, flags=re.IGNORECASE):
            val = float(m.group(1))
            if "ms" in p or "millisecond" in p:
                dur_s = val / 1000.0
            else:
                dur_s = val
            break
    if dur_s is None:
        if m := re.search(r"(\d+(?:\.\d+)?)ms\b", line):
            dur_s = float(m.group(1)) / 1000.0
    return ts, dur_s


# --- Parser with context tracking ---
def parse_log_with_timing_ms(lines: List[str]):
    parsed = []
    last_hnsw_build = None
    last_ivf_build = None

    for i, line in enumerate(lines, start=1):
        raw = line.rstrip("\n")
        ts, dur_s = extract_time_info(raw)
        dur_ms_str = None
        if dur_s is not None:
            dur_ms = dur_s * 1000.0
            dur_ms_str = f"{dur_ms:.3f} ms"

        # HNSW build
        if m := re.search(r"hnsw_build_(\w+)_(\d+)_(\d+)", raw):
            ent = HNSWBuild(raw, i, ts, dur_ms_str, m.group(1), int(m.group(2)), int(m.group(3)))
            last_hnsw_build = ent
            parsed.append(ent)
            continue

        # HNSW query
        if m := re.search(r"hnsw_query_(\d+)_(\w+)\[(\d+)\]\[(\d+)\]", raw):
            if last_hnsw_build:
                ent = HNSWQueryEnhanced(raw, i, ts, dur_ms_str,
                                        ef_search=int(m.group(1)),
                                        dist_func=m.group(2),
                                        topk=int(m.group(3)),
                                        query_count=int(m.group(4)),
                                        max_cons=last_hnsw_build.max_cons,
                                        ef_construction=last_hnsw_build.ef_construction)
            else:
                ent = HNSWQueryEnhanced(raw, i, ts, dur_ms_str,
                                        ef_search=int(m.group(1)),
                                        dist_func=m.group(2),
                                        topk=int(m.group(3)),
                                        query_count=int(m.group(4)),
                                        max_cons=None,
                                        ef_construction=None)
            parsed.append(ent)
            continue

        # HNSW recall
        if m := re.search(r"hnsw_recall_(\d+)_(\d+)_(\d+)\[(\d+)\]", raw):
            parsed.append(HNSWRecall(raw, i, ts, dur_ms_str,
                                     int(m.group(1)), int(m.group(2)),
                                     int(m.group(3)), int(m.group(4))))
            continue

        # IVF build
        if m := re.search(r"ivf_build_(\w+)_(\d+)", raw):
            ent = IVFBuild(raw, i, ts, dur_ms_str, m.group(1), int(m.group(2)))
            last_ivf_build = ent
            parsed.append(ent)
            continue

        # IVF query
        if m := re.search(r"ivf_query_(\d+)_(\w+)\[(\d+)\]\[(\d+)\]", raw):
            if last_ivf_build:
                ent = IVFQueryEnhanced(raw, i, ts, dur_ms_str,
                                       n_probes=int(m.group(1)),
                                       dist_func=m.group(2),
                                       topk=int(m.group(3)),
                                       query_count=int(m.group(4)),
                                       ivf_list_count=last_ivf_build.ivf_list_count)
            else:
                ent = IVFQueryEnhanced(raw, i, ts, dur_ms_str,
                                       n_probes=int(m.group(1)),
                                       dist_func=m.group(2),
                                       topk=int(m.group(3)),
                                       query_count=int(m.group(4)),
                                       ivf_list_count=None)
            parsed.append(ent)
            continue

        # Exact search
        if m := re.search(r"exact_search_(\w+)\[(\d+)\]\[(\d+)\]", raw):
            parsed.append(ExactSearch(raw, i, ts, dur_ms_str,
                                      m.group(1), int(m.group(2)), int(m.group(3))))
            continue

        # IVF recall
        if m := re.search(r"ivf_recall_(\d+)_(\d+)\[(\d+)\]", raw):
            parsed.append(IVFRecall(raw, i, ts, dur_ms_str,
                                    int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue

    return parsed


# --- Usage Example ---
if __name__ == "__main__":
    log_path = "log.log"
    with open(log_path, "r") as f:
        lines = f.readlines()

    parsed = parse_log_with_timing_ms(lines)

    df = pd.DataFrame([asdict(p) | {"type": type(p).__name__} for p in parsed])
    df = df[["line_no", "timestamp", "duration", "type"] +
            [c for c in df.columns if c not in ["line_no", "timestamp", "duration", "type"]]]

    df.to_csv("parsed_log_with_timing_ms.csv", index=False)
    # Show full DataFrame without truncation
    pd.set_option("display.max_rows", None)      # Show all rows
    pd.set_option("display.max_columns", None)   # Show all columns
    pd.set_option("display.width", 0)            # Prevent line wrapping
    pd.set_option("display.max_colwidth", None)  # Show full cell content

    print(df.to_string(index=False))

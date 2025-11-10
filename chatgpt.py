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
    recall:dict[str,float]=None

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
    recall:dict[str,float]=None


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
            recall=eval(raw.split(']')[-1])
            parsed.append(HNSWRecall(raw, i, ts, dur_ms_str,
                                     int(m.group(1)), int(m.group(2)),
                                     int(m.group(3)), int(m.group(4)),recall))
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
            recall=eval(raw.split(']')[-1])
            parsed.append(IVFRecall(raw, i, ts, dur_ms_str,
                                    int(m.group(1)), int(m.group(2)), int(m.group(3)),recall))
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
# import matplotlib.pyplot as plt

# # --- Filter for IVFQueryEnhanced, dist_func=l2, topk=100 ---
# ivf_df = df[df["type"] == "IVFQueryEnhanced"].copy()
# ivf_df = ivf_df[ivf_df["duration"].notnull()]
# ivf_df["duration_ms"] = ivf_df["duration"].str.replace(" ms", "", regex=False).astype(float)

# ivf_l2_topk100 = ivf_df[(ivf_df["dist_func"] == "l2") & (ivf_df["topk"] == 100)]

# avg_times = (
#     ivf_l2_topk100
#     .groupby("n_probes", as_index=False)["duration_ms"]
#     .mean()
#     .sort_values("n_probes")
# )

# print("\nAverage IVF Query Times (L2, topk=100):")
# print(avg_times)

# plt.figure(figsize=(7, 5))
# plt.plot(avg_times["n_probes"], avg_times["duration_ms"], marker="o", linestyle="-", linewidth=2)
# plt.title("IVF Query Time vs n_probes (L2, topk=100)")
# plt.xlabel("n_probes")
# plt.ylabel("Average Duration (ms)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("ivf_query_time_vs_nprobes.png")
# print("Plot saved as ivf_query_time_vs_nprobes.png")
# --- HNSW Query Timing Analysis ---
# import matplotlib.pyplot as plt

# # # Filter to only HNSW queries
# hnsw_df = df[df["type"] == "HNSWQueryEnhanced"].copy()

# # Drop entries without duration
# hnsw_df = hnsw_df[hnsw_df["duration"].notnull()]

# # Convert duration "xxx ms" → numeric
# hnsw_df["duration_ms"] = hnsw_df["duration"].str.replace(" ms", "", regex=False).astype(float)

# # Filter for L2 distance and topk = 100
# hnsw_l2_topk100 = hnsw_df[(hnsw_df["dist_func"] == "l2") & (hnsw_df["topk"] == 100)]

# # Group by ef_search and compute average time
# avg_hnsw = (
#     hnsw_l2_topk100
#     .groupby("ef_search", as_index=False)["duration_ms"]
#     .mean()
#     .sort_values("ef_search")
# )

# print("\nAverage HNSW Query Times (L2, topk=100):")
# print(avg_hnsw)

# # --- Plot ---
# plt.figure(figsize=(7, 5))
# plt.plot(avg_hnsw["ef_search"], avg_hnsw["duration_ms"], marker="o", linestyle="-", linewidth=2)
# plt.title("HNSW Query Time vs ef_search (L2, topk=100)")
# plt.xlabel("ef_search")
# plt.ylabel("Average Duration (ms)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("hnsw_query_time_vs_efsearch.png")
# print("Plot saved as hnsw_query_time_vs_efsearch.png")
# # plt.show()  # uncomment if you want to see the window
import matplotlib.pyplot as plt

# --- Extract IVF Recall Entries ---
ivf_recalls = [p for p in parsed if isinstance(p, IVFRecall)]

# Collect recall values safely (no eval)
rows = []
for p in ivf_recalls:
    if not p.recall:
        continue
    for key, val in p.recall.items():
        try:
            list_size, dist_func = key.split("_")
            list_size = int(list_size)
            rows.append({
                "ivf_list_count": p.ivf_list_count,
                "n_probes": p.n_probes,
                "topk": p.query_count,
                "dist_func": dist_func,
                "recall_value": val
            })
        except Exception:
            continue

recall_df = pd.DataFrame(rows)

# --- Filter for L2 distance and topk = 100 ---
filtered = recall_df[(recall_df["dist_func"] == "l2") & (recall_df["topk"] == 100)]

# --- Compute Average Recall by n_probes ---
avg_recall = (
    filtered.groupby("n_probes", as_index=False)["recall_value"]
    .mean()
    .sort_values("n_probes")
)

print("\nAverage IVF Recall (L2, topk=100) vs n_probes:")
print(avg_recall)

# --- Plot ---
plt.figure(figsize=(7, 5))
plt.plot(avg_recall["n_probes"], avg_recall["recall_value"], marker="o", linewidth=2)
plt.title("IVF Recall (L2, topk=100) vs n_probes")
plt.xlabel("n_probes")
plt.ylabel("Average Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig("ivf_recall_vs_nprobes_l2.png")
print("Plot saved as ivf_recall_vs_nprobes_l2.png")
# plt.show()  # you can uncomment if you want to display interactively

# --- Extract HNSW Recall Entries ---
hnsw_recalls = [p for p in parsed if isinstance(p, HNSWRecall)]

# Collect recall values from the dict safely
rows = []
for p in hnsw_recalls:
    if not p.recall:
        continue
    for key, val in p.recall.items():
        try:
            list_size, dist_func = key.split("_")
            list_size = int(list_size)
            rows.append({
                "ef_search": p.ef_search,
                "max_cons": p.max_cons,
                "ef_construction": p.ef_construction,
                "topk": p.query_count,
                "dist_func": dist_func,
                "recall_value": val
            })
        except Exception:
            continue

hnsw_recall_df = pd.DataFrame(rows)

# --- Filter for ef_construction = 32, dist_func = "l2", and topk = 100 ---
filtered = hnsw_recall_df[
    (hnsw_recall_df["ef_construction"] == 32)
    & (hnsw_recall_df["dist_func"] == "l2")
    & (hnsw_recall_df["topk"] == 100)
]

# --- Compute Average Recall by ef_search ---
avg_recall = (
    filtered.groupby("ef_search", as_index=False)["recall_value"]
    .mean()
    .sort_values("ef_search")
)

print("\nAverage HNSW Recall (efc=32, L2, topk=100) vs ef_search:")
print(avg_recall)

# --- Plot ---
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.plot(avg_recall["ef_search"], avg_recall["recall_value"], marker="o", linewidth=2)
plt.title("HNSW Recall (efc=32, L2, topk=100) vs ef_search")
plt.xlabel("ef_search")
plt.ylabel("Average Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig("hnsw_recall_vs_efsearch_l2.png")
print("Plot saved as hnsw_recall_vs_efsearch_l2.png")
# plt.show()  # Uncomment if you want to view interactively
import pandas as pd
import matplotlib.pyplot as plt

# ---------- IVF Recall vs Time ----------
ivf_queries = [p for p in parsed if isinstance(p, IVFQueryEnhanced)]
ivf_query_df = pd.DataFrame([
    {"n_probes": p.n_probes,
     "ivf_list_count": p.ivf_list_count,
     "dist_func": p.dist_func,
     "topk": p.topk,
     "duration_ms": float(p.duration.replace(" ms", "")) if p.duration else None}
    for p in ivf_queries if p.duration
])

ivf_recall_rows = []
for p in [p for p in parsed if isinstance(p, IVFRecall)]:
    if not p.recall:
        continue
    for key, val in p.recall.items():
        try:
            list_size, dist_func = key.split("_")
            list_size = int(list_size)
            ivf_recall_rows.append({
                "ivf_list_count": p.ivf_list_count,
                "n_probes": p.n_probes,
                "topk": p.query_count,
                "dist_func": dist_func,
                "recall_value": val
            })
        except Exception:
            continue

ivf_recall_df = pd.DataFrame(ivf_recall_rows)

# Filter (L2, topk=100)
ivf_recall_filtered = ivf_recall_df[(ivf_recall_df["dist_func"] == "l2") & (ivf_recall_df["topk"] == 100)]
ivf_time_filtered = ivf_query_df[(ivf_query_df["dist_func"] == "l2") & (ivf_query_df["topk"] == 100)]

ivf_recall_avg = ivf_recall_filtered.groupby("n_probes", as_index=False)["recall_value"].mean()
ivf_time_avg = ivf_time_filtered.groupby("n_probes", as_index=False)["duration_ms"].mean()

ivf_merged = pd.merge(ivf_recall_avg, ivf_time_avg, on="n_probes", how="inner").sort_values("n_probes")
ivf_merged["algo"] = "IVF"

# ---------- HNSW Recall vs Time ----------
hnsw_queries = [p for p in parsed if isinstance(p, HNSWQueryEnhanced)]
hnsw_query_df = pd.DataFrame([
    {"ef_search": p.ef_search,
     "max_cons": p.max_cons,
     "ef_construction": p.ef_construction,
     "dist_func": p.dist_func,
     "topk": p.topk,
     "duration_ms": float(p.duration.replace(" ms", "")) if p.duration else None}
    for p in hnsw_queries if p.duration
])

hnsw_recall_rows = []
for p in [p for p in parsed if isinstance(p, HNSWRecall)]:
    if not p.recall:
        continue
    for key, val in p.recall.items():
        try:
            list_size, dist_func = key.split("_")
            list_size = int(list_size)
            hnsw_recall_rows.append({
                "ef_search": p.ef_search,
                "max_cons": p.max_cons,
                "ef_construction": p.ef_construction,
                "topk": p.query_count,
                "dist_func": dist_func,
                "recall_value": val
            })
        except Exception:
            continue

hnsw_recall_df = pd.DataFrame(hnsw_recall_rows)

# Filter (efc=32, L2, topk=100)
hnsw_recall_filtered = hnsw_recall_df[
    (hnsw_recall_df["ef_construction"] == 32)
    & (hnsw_recall_df["dist_func"] == "l2")
    & (hnsw_recall_df["topk"] == 100)
]
hnsw_time_filtered = hnsw_query_df[
    (hnsw_query_df["ef_construction"] == 32)
    & (hnsw_query_df["dist_func"] == "l2")
    & (hnsw_query_df["topk"] == 100)
]

hnsw_recall_avg = hnsw_recall_filtered.groupby("ef_search", as_index=False)["recall_value"].mean()
hnsw_time_avg = hnsw_time_filtered.groupby("ef_search", as_index=False)["duration_ms"].mean()

hnsw_merged = pd.merge(hnsw_recall_avg, hnsw_time_avg, on="ef_search", how="inner").sort_values("ef_search")
hnsw_merged["algo"] = "HNSW"

# ---------- Combined Plot ----------
plt.figure(figsize=(8, 6))

plt.plot(ivf_merged["duration_ms"], ivf_merged["recall_value"], marker="o", linewidth=2, label="IVF")
plt.plot(hnsw_merged["duration_ms"], hnsw_merged["recall_value"], marker="s", linewidth=2, label="HNSW")

plt.title("Recall vs Query Time (L2, topk=100)")
plt.xlabel("Average Query Time (ms)")
plt.ylabel("Average Recall")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("recall_vs_time_l2.png")

print("✅ Combined plot saved as recall_vs_time_l2.png")

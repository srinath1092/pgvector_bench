import re
from collections import defaultdict
import json

def parse_logs(log_content):
    """
    Parses log content to find average time and memory for different
    ivf_build_l2 sizes.
    """

    stats = defaultdict(lambda: {
        'times': [],
        'memories': [],
        'time_sum': 0.0,
        'mem_sum': 0.0,
        'count': 0
    })

    # Regex patterns to capture the size, time, and memory
    # \[ivf_build_l2_(\d+)\]: Matches the prefix and captures the size (e.g., 120)
    # Time (\S+): Matches "Time" and captures the time value
    # Memory (\S+) kB: Matches "Memory", captures the memory value, and expects "kB"
    time_pattern = re.compile(r'\[hnsw_build_l2_16_(\d+)\] Time (\S+)')
    mem_pattern = re.compile(r'\[hnsw_build_l2_16_(\d+)\] Memory (\S+) kB')

    for line in log_content.splitlines():
        time_match = time_pattern.search(line)
        if time_match:
            size = time_match.group(1)
            time_val = float(time_match.group(2))
            
            stats[size]['times'].append(time_val)
            stats[size]['time_sum'] += time_val
            # We increment count only on 'Time' lines, assuming each entry
            # has both a Time and a Memory line.
            stats[size]['count'] += 1
            continue # Move to the next line

        mem_match = mem_pattern.search(line)
        if mem_match:
            size = mem_match.group(1)
            mem_val = float(mem_match.group(2))
            
            stats[size]['memories'].append(mem_val)
            stats[size]['mem_sum'] += mem_val

    # Calculate averages and format the results
    results = {}
    for size, data in stats.items():
        if data['count'] > 0:
            avg_time = data['time_sum'] / data['count']
            # Assuming memory entries always correspond to time entries
            avg_memory = data['mem_sum'] / data['count'] 
            
            results[size] = {
                'average_time_seconds': avg_time,
                'average_memory_kb': avg_memory,
                'entry_count': data['count']
            }

    return results

# --- Example Usage ---

if __name__ == "__main__":

    # --- To read from a file instead ---
    try:
        with open('log.log', 'r') as f:
            log_data = f.read()
    except FileNotFoundError:
        print("Log file not found. Using example data.")
        # Keep using the hardcoded log_data as a fallback
    
    parsed_data = parse_logs(log_data)

    # Pretty print the results
    for size, data in parsed_data.items():
        print(f"\nResults for size: {size}")
        print(f"  Entry Count: {data['entry_count']}")
        print(f"  Avg. Time:   {data['average_time_seconds']:.6f} s")
        print(f"  Avg. Memory: {data['average_memory_kb']:.2f} kB")
"""
Calculate the complexity score for each sample based on metadata from the generated report.json, and save the results as a CSV file.
"""
import os
import json
import pandas as pd

'''
Complexity Calculation Rules (two dimensions, four levels)
1. Node Count Score
- ≤40 → 1 point
- 41–120 → 2 points
- 121–250 → 3 points
- >250 → 4 points

2. Maximum Nesting Depth Score
- ≤3 levels → 1 point
- 4-5 levels → 2 points
- 6-7 levels → 3 points
- >7 levels → 4 points

3. Comprehensive Score → Complexity Level
| Total Score Range | Level |
'''
def score_node_count(n):
    if n <= 40:
        return 1
    elif n <= 120:
        return 2
    elif n <= 250:
        return 3
    else:
        return 4

def score_depth(d):
    if d <= 3:
        return 1
    elif d <= 5:
        return 2
    elif d <= 7:
        return 3
    else:
        return 4


def compute_complexity(statistics):
    node_score = score_node_count(statistics.get("node_counts", 0))
    depth_score = score_depth(statistics.get("max_depth", 0))

    total = node_score + depth_score

    if total <= 3:
        complexity = "low"
    elif total <= 5:
        complexity = "mid"
    elif total <= 7:
        complexity = "high"
    else:
        complexity = "hard"

    return {
        "node_score": node_score,
        "depth_score": depth_score,
        "total_score": total,
        "complexity": complexity
    }


# -------------------------
# Main program
# -------------------------
def process_reports(base_dir, output_csv="complexity_results.csv"):
    results = []
    match_count = 0
    total_count = 0

    for filekey in os.listdir(base_dir):
        report_path = os.path.join(base_dir, filekey, "report.json")
        if os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            stats = report.get("statistics", {})
            
            res = compute_complexity(stats)

            # Add ID information
            res["file_key"] = report.get("file_key", filekey)
            res["node_id"] = report.get("node_id")
            res["node_counts"] = stats.get("node_counts", 0)
            res["max_depth"] = stats.get("max_depth", 0)
            res["annotation_complexity"] = report.get("annotation", {}).get("complexity")

            # Compare for consistency
            res["match"] = (str(res["complexity"]) == str(res["annotation_complexity"]))
            if res["match"]:
                match_count += 1
            total_count += 1

            results.append(res)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

    # Calculate match rate
    match_rate = match_count / total_count if total_count > 0 else 0
    print(f"📊 Match rate: {match_rate:.2%} ({match_count}/{total_count})")

    return df, match_rate


# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    from ...configs.paths import DATA_TEST_DIR
    base_dir = DATA_TEST_DIR
    df, match_rate = process_reports(base_dir, output_csv="complexity_results.csv")
    print(df.head())

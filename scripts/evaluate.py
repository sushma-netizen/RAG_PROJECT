import argparse, json, csv, math
from pathlib import Path

def load_ground_truth(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize keys to str
        return {str(k): set(map(str.lower, v)) for k, v in data.items()}
    elif p.suffix.lower() == ".csv":
        # Expect columns: query_number, answers  (answers are ; separated)
        truth = {}
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row["query_number"]).strip()
                answers = [a.strip().lower() for a in row["answers"].split(";") if a.strip()]
                truth[qid] = set(answers)
        return truth
    else:
        raise ValueError("truth_file must be .json or .csv")

def dcg_at_k(relevances):
    """relevances: list of 1/0 for ranks 1..k"""
    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        # log2(i+1) = math.log2(i+1)
        dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(pred, gt, k):
    """Binary relevance: any predicted file in gt is relevant."""
    topk = pred[:k]
    rels = [1 if x in gt else 0 for x in topk]
    dcg = dcg_at_k(rels)
    ideal_rels = [1] * min(k, len(gt))
    idcg = dcg_at_k(ideal_rels)
    return (dcg / idcg) if idcg > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Folder with system outputs as <qid>.json")
    ap.add_argument("--truth_file", required=True, help="Path to ground truth (.json or .csv)")
    ap.add_argument("--k", type=int, default=5, help="k for @k metrics")
    ap.add_argument("--out_dir", default=None, help="Where to write eval results")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    truth = load_ground_truth(args.truth_file)
    k = args.k
    out_dir = Path(args.out_dir) if args.out_dir else (pred_dir.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    qids_evaluated = 0
    p_sum = r_sum = ndcg_sum = 0.0

    for pred_path in sorted(pred_dir.glob("*.json")):
        qid = pred_path.stem  # filename without .json
        if qid not in truth:
            # skip queries without ground truth
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_json = json.load(f)

        # read predicted list
        pred_list = pred_json.get("response", []) or []
        # compare using lowercase basenames
        pred_norm = [Path(x).name.lower() for x in pred_list]
        gt = truth[qid]

        hits = len(set(pred_norm[:k]) & gt)
        precision = hits / max(k, 1)
        recall = hits / max(len(gt), 1)
        ndcg = ndcg_at_k(pred_norm, gt, k)

        rows.append({
            "query_number": qid,
            "num_gt": len(gt),
            "num_predicted": len(pred_norm),
            "hits@k": hits,
            f"precision@{k}": round(precision, 4),
            f"recall@{k}": round(recall, 4),
            f"ndcg@{k}": round(ndcg, 4),
            "ground_truth_files": "; ".join(sorted(gt)),
            "predicted_topk": "; ".join(pred_norm[:k]),
        })

        p_sum += precision
        r_sum += recall
        ndcg_sum += ndcg
        qids_evaluated += 1

    # Write detailed CSV
    detailed_csv = out_dir / "per_query_metrics.csv"
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_number","num_gt","num_predicted","hits@k",
                      f"precision@{k}", f"recall@{k}", f"ndcg@{k}",
                      "ground_truth_files","predicted_topk"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write summary
    summary_txt = out_dir / "summary.txt"
    if qids_evaluated > 0:
        p = p_sum / qids_evaluated
        r = r_sum / qids_evaluated
        nd = ndcg_sum / qids_evaluated
    else:
        p = r = nd = 0.0

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Queries evaluated: {qids_evaluated}\n")
        f.write(f"Precision@{k}: {p:.4f}\n")
        f.write(f"Recall@{k}: {r:.4f}\n")
        f.write(f"NDCG@{k}: {nd:.4f}\n")

    print(f"\nWrote: {detailed_csv}")
    print(f"Wrote: {summary_txt}\n")

if __name__ == "__main__":
    main()


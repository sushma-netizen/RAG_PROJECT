#!/usr/bin/env python3
# search_and_export.py (updated, with verbose/debug prints)
# Usage:
# python .\scripts\search_and_export.py --index_dir "D:\RAG_Project\index" --queries "D:\RAG_Project\queries.csv" --out_dir "D:\RAG_Project\outputs" --topk 5 --search_k 50 --startup "MyStartup" --verbose

import os
import json
import argparse
import shutil
import traceback
from collections import OrderedDict

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

def load_metadata(meta_path):
    id2meta = {}
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                id2meta[int(obj["id"])] = obj
            except Exception:
                # skip bad lines but continue
                continue
    return id2meta

def pick_query_columns(df):
    # prefer columns named 'query' or 'question' (case-insensitive)
    cols = list(df.columns)
    q_col = None
    id_col = None
    for c in cols:
        if c.lower() in ("id","qid","query_id","serial"):
            id_col = c
            break
    if id_col is None:
        id_col = cols[0]  # fallback
    for c in cols:
        if c.lower() in ("query","question","text"):
            q_col = c
            break
    if q_col is None:
        if len(cols) >= 2:
            q_col = cols[1]
        else:
            q_col = cols[0]
    return id_col, q_col

def main(args):
    try:
        print("=== search_and_export.py starting ===")
        print("Index dir:", args.index_dir)
        print("Queries file:", args.queries)
        print("Output dir:", args.out_dir)
        print("TopK sources per query:", args.topk, "Search (chunk) k:", args.search_k)
        print("Model:", args.model_name)
        if args.verbose:
            print("Verbose mode ON")

        index_path = os.path.join(args.index_dir, "faiss_index.bin")
        meta_path = os.path.join(args.index_dir, "metadata.jsonl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at: {meta_path}")

        # Load model
        print("Loading embedding model (this may take a moment)...")
        model = SentenceTransformer(args.model_name)

        # Load index and metadata
        print("Loading FAISS index...")
        index = faiss.read_index(index_path)
        print("Index loaded. ntotal =", index.ntotal)
        print("Loading metadata...")
        id2meta = load_metadata(meta_path)
        print("Metadata entries:", len(id2meta))

        # Prepare outputs folder
        os.makedirs(args.out_dir, exist_ok=True)

        # Read queries CSV
        if not os.path.exists(args.queries):
            raise FileNotFoundError(f"Queries file not found: {args.queries}")

        df = pd.read_csv(args.queries, encoding="utf-8")
        if df.shape[0] == 0:
            print("Queries CSV is empty. Nothing to do.")
            return

        id_col, q_col = pick_query_columns(df)
        if args.verbose:
            print("Detected id column:", id_col, " query column:", q_col)

        written = 0
        # iterate queries
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
            try:
                qid = str(row[id_col])
                query_text = str(row[q_col])
            except Exception as e:
                if args.verbose:
                    print("Skipping a row due to read error:", e)
                continue

            if query_text.strip() == "":
                if args.verbose:
                    print(f"Skipping empty query for id {qid}")
                continue

            # embed & search
            q_emb = model.encode([query_text], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            D, I = index.search(q_emb, k=args.search_k)

            # map chunk ids -> source files (dedupe, keep order)
            unique_sources = []
            seen = set()
            if I is None or len(I) == 0:
                if args.verbose:
                    print(f"No hits for query id {qid}")
            else:
                for idx in I[0]:
                    if int(idx) < 0:
                        continue
                    meta = id2meta.get(int(idx))
                    if not meta:
                        continue
                    src = meta.get("source")
                    if not src:
                        continue
                    src_basename = os.path.basename(src)
                    if src_basename not in seen:
                        unique_sources.append(src_basename)
                        seen.add(src_basename)
                    if len(unique_sources) >= args.topk:
                        break

            # prepare json payload
            out_obj = {
                "query": query_text,
                "response": unique_sources
            }
            out_path = os.path.join(args.out_dir, f"{qid}.json")
            with open(out_path, "w", encoding="utf-8") as fo:
                json.dump(out_obj, fo, ensure_ascii=False, indent=2)
            written += 1
            if args.verbose:
                print(f"Wrote {out_path} -> {len(unique_sources)} sources")

        print(f"Finished queries. Total JSON files written: {written}")

        # zip outputs
        zip_base = os.path.join(os.path.dirname(args.out_dir), f"{args.startup}_PS4")
        zip_file = shutil.make_archive(zip_base, 'zip', args.out_dir)
        print("Created zip:", zip_file)
        print("=== Done ===")

    except Exception as e:
        print("ERROR during execution:")
        traceback.print_exc()
        print("Exiting with failure.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--search_k", type=int, default=50)
    parser.add_argument("--startup", type=str, default="myStartup")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2")
    parser.add_argument("--verbose", action="store_true", help="Turn on verbose logging")
    args = parser.parse_args()
    main(args)

# build_index.py
# Batch-index text/JSON/TSV files into a FAISS index with metadata (safe for large corpora)
# Usage example (PowerShell):
#   python .\scripts\build_index.py --data_dir "D:\RAG_Project\data" --index_dir "D:\RAG_Project\index" --batch_size 512

import os, glob, json, argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

def chunk_text(text, chunk_size=1000, overlap=200):
    text = text.strip()
    if len(text) <= chunk_size:
        yield (0, text)
        return
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield (start, text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap

def extract_text_from_file(path):
    # Try JSON, TSV/CSV, or plain text
    name = os.path.basename(path).lower()
    try:
        if name.endswith(".json"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                j = json.load(f)
            # heuristics: combine meaningful fields
            pieces = []
            for k in ("question","answer","title","snippet","text","body"):
                if isinstance(j.get(k), str):
                    pieces.append(j.get(k))
            # if it looks like search_results (list of dicts)
            if "search_results" in j and isinstance(j["search_results"], list):
                for item in j["search_results"]:
                    if isinstance(item, dict):
                        for key in ("snippet","title"):
                            if key in item and isinstance(item[key], str):
                                pieces.append(item[key])
            # fallback: stringify whole json
            if not pieces:
                pieces.append(json.dumps(j))
            return " \n ".join(pieces)
        else:
            # TSV/CSV or plain text: just read
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return ""

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main(args):
    ensure_dir(args.index_dir)
    model = SentenceTransformer(args.model_name)
    dim = model.get_sentence_embedding_dimension()
    # Base FAISS index: cosine similarity via normalized vectors + IndexFlatIP
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)

    meta_path = os.path.join(args.index_dir, "metadata.jsonl")
    index_path = os.path.join(args.index_dir, "faiss_index.bin")
    processed_path = os.path.join(args.index_dir, "processed_files.txt")

    # Resume logic: if index already exists, load it and find next id
    next_id = 0
    processed_files = set()
    if os.path.exists(meta_path):
        print("Found existing metadata; reading to determine next_id...")
        max_id = -1
        with open(meta_path, "r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    obj = json.loads(line)
                    if obj.get("id", -1) > max_id:
                        max_id = obj["id"]
                except:
                    continue
        next_id = max_id + 1
    if os.path.exists(processed_path):
        with open(processed_path, "r", encoding="utf-8") as pf:
            processed_files = set([l.strip() for l in pf if l.strip()])

    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_path)

    # gather files (top-level only, adjust walk if needed)
    all_files = []
    for root, _, files in os.walk(args.data_dir):
        for fname in files:
            if fname.lower().endswith((".json",".txt",".tsv",".csv")):
                all_files.append(os.path.join(root, fname))
    all_files.sort()
    print(f"Total candidate files found: {len(all_files)}")

    batch_texts = []
    batch_meta = []
    saved = 0
    with open(meta_path, "a", encoding="utf-8") as meta_f, open(processed_path, "a", encoding="utf-8") as proc_f:
        for file_path in tqdm(all_files, desc="Files"):
            if os.path.abspath(file_path) in processed_files:
                continue
            text = extract_text_from_file(file_path)
            if not text:
                # still mark as processed to skip next time
                proc_f.write(os.path.abspath(file_path) + "\n")
                proc_f.flush()
                continue
            # chunk
            for start, chunk in chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap):
                # small cleanup
                chunk = " ".join(chunk.split())
                if len(chunk) < 20:
                    continue
                meta = {
                    "id": next_id,
                    "source": os.path.abspath(file_path),
                    "start": start,
                    "excerpt": chunk[: args.excerpt_len]
                }
                batch_texts.append(chunk)
                batch_meta.append(meta)
                next_id += 1

                # when batch full -> encode + add to FAISS + write metadata
                if len(batch_texts) >= args.batch_size:
                    embs = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, batch_size=args.encode_batch_size)
                    # normalize for cosine (inner product on normalized vectors)
                    faiss.normalize_L2(embs)
                    ids = np.array([m["id"] for m in batch_meta], dtype=np.int64)
                    index.add_with_ids(embs.astype(np.float32), ids)
                    # write metadata lines
                    for m in batch_meta:
                        meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    meta_f.flush()
                    # update processed (we mark file processed only after all its chunks added)
                    # but here we don't know boundaries; to be safe we write processed file at end of file loop
                    # reset batch
                    batch_texts = []
                    batch_meta = []
                    saved += len(ids)
                    if saved % (args.save_every) == 0:
                        print(f"Saving index to {index_path} (checkpoint).")
                        faiss.write_index(index, index_path)

            # finished this file: mark processed
            proc_f.write(os.path.abspath(file_path) + "\n")
            proc_f.flush()

        # handle remaining batch
        if batch_texts:
            embs = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, batch_size=args.encode_batch_size)
            faiss.normalize_L2(embs)
            ids = np.array([m["id"] for m in batch_meta], dtype=np.int64)
            index.add_with_ids(embs.astype(np.float32), ids)
            for m in batch_meta:
                meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")
            meta_f.flush()
            saved += len(ids)

        # final save
        print(f"Final save of FAISS index to {index_path} (total chunks added: {saved})")
        faiss.write_index(index, index_path)
    print("Indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder with input files")
    parser.add_argument("--index_dir", required=True, help="Folder to store index & metadata")
    parser.add_argument("--batch_size", type=int, default=512, help="Number of chunks to embed before adding to FAISS")
    parser.add_argument("--encode_batch_size", type=int, default=32, help="SentenceTransformer internal batch size")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Characters per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--excerpt_len", type=int, default=300, help="How many chars to store as excerpt in metadata")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2", help="Sentence-Transformers model")
    parser.add_argument("--save_every", type=int, default=4096, help="Checkpoint save frequency (chunks)")
    args = parser.parse_args()
    main(args)

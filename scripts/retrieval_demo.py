from sentence_transformers import SentenceTransformer
import faiss
import os, glob

# Load embedding model
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Collect text files
data_dir = r"D:\RAG_Project\data"
files = glob.glob(os.path.join(data_dir, "*"))
documents = []

for f in files:
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            text = fp.read()
            documents.append((f, text))
    except Exception as e:
        print(f"Skipping {f}: {e}")

print(f"Loaded {len(documents)} documents")

# Embed documents
texts = [doc[1][:1000] for doc in documents]  # limit to first 1000 chars
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("FAISS index built with", index.ntotal, "documents")

# --- Interactive search loop ---
while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=3)

    print("\nTop results:")
    for rank, idx in enumerate(I[0], start=1):
        print(f"\nRank {rank} | File: {documents[idx][0]}")
        print(documents[idx][1][:300], "...")

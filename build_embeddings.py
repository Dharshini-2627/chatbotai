import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient

CSV_PATH = "activities.csv"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sustainability"

print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

print("Columns found:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

print("\nLoading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("./all-MiniLM-L6-v2")

docs = df["Activity"].astype(str).tolist()

# Convert to dict and ensure all values are strings or numbers
metadatas = []
for _, row in df.iterrows():
    meta = {}
    for col in df.columns:
        value = row[col]
        # Convert NaN to empty string, keep numbers as float/int
        if pd.isna(value):
            meta[col] = ""
        elif isinstance(value, (int, float)):
            meta[col] = float(value)
        else:
            meta[col] = str(value)
    metadatas.append(meta)

print("\nSample metadata:")
print(metadatas[0])

print("\nComputing embeddings...")
embeddings = embed_model.encode(docs, show_progress_bar=True, convert_to_numpy=True).tolist()

print("Starting Chroma client...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    collection = client.get_collection(COLLECTION_NAME)
    print("Collection exists, deleting and recreating...")
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(name=COLLECTION_NAME)

print("Adding documents to collection...")
collection.add(
    documents=docs,
    metadatas=metadatas,
    ids=[str(i) for i in range(len(docs))],
    embeddings=embeddings
)

print(f"\n✅ Done! Added {len(docs)} documents to ChromaDB at {CHROMA_DIR}")
print("\nVerifying storage...")
test_result = collection.query(
    query_embeddings=[embeddings[0]],
    n_results=1,
    include=["documents", "metadatas"]
)
print("Test query metadata:", test_result["metadatas"][0][0])



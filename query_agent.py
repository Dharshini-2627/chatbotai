from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import math
from chromadb import PersistentClient
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "sustainability"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path=CHROMA_DIR)
persist_directory=CHROMA_DIR
collection = client.get_collection(COLLECTION_NAME)
def retrieve(query, k=4):
    q_emb = embed_model.encode([query])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas","distances"])
    return res
print("Loading LLM (this may take time)...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
def build_prompt(query, retrieved):
    docs = retrieved["documents"][0]
    metas = retrieved["metadatas"][0]
    block = []
    for d,m in zip(docs, metas):
        block.append(f"- {d} | emission: {m.get('Avg_CO2_Emission(kg/day)','?')} kg/day | {m.get('Category','')}")
        # include tip if present in metadata
        if m.get("Tip"):
            block.append(f"  Tip: {m['Tip']}")
    retrieved_text = "\n".join(block)
    prompt = f"""You are a concise sustainability assistant. Use the retrieved info below and answer the user question concisely with:
1) short current-emission estimate (use retrieved Avg_CO2_Emission if matches).
2) 2–4 actionable steps with quick CO2 impact estimates if possible.
3) one long-term suggestion.

Retrieved info:
{retrieved_text}

User question:
{query}

Answer:"""
    return prompt

def answer_query(query, top_k=4):
    retrieved = retrieve(query, k=top_k)
    prompt = build_prompt(query, retrieved)
    out = pipe(prompt)[0]["generated_text"]
    return out

if __name__ == "__main__":
    q = input("Type your CO2 question (e.g. 'I drive 20 km daily using a petrol car...'):\n")
    print("\n--- thinking ---\n")
    print(answer_query(q, top_k=4))

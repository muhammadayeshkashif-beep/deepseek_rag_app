import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import streamlit as st

# ====== CONFIG ======
DB_DIR = "chroma_db"  # local folder for your vector DB
COLLECTION_NAME = "social_posts"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEN_MODEL = "deepseek/deepseek-r1:free"

SYSTEM_PROMPT = """
You are 'CMO-DeepSeek', a senior digital marketing thought leader and content strategist.
Write with authority and clarity about digital marketing strategy, paid search, paid social, analytics, and ROI.
Audience lenses to consider when asked: CMO (growth/efficiency), CEO (revenue/advantage), CFO (ROI/CAC/LTV/budgets),
Director of Digital (execution/scaling).
Style rules: concise, data-aware, practical frameworks, executive tone, end with 1 crisp takeaway or CTA.
Use the retrieved posts below strictly as STYLE inspiration (tone, cadence, structure)—do not copy them verbatim.
"""

# ====== INIT: embeddings + vector DB + LLM client ======
embedder = SentenceTransformer(EMBED_MODEL_NAME)

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

chroma = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)


# ====== HELPERS ======
def embed_texts(texts):
    return embedder.encode(texts, normalize_embeddings=True).tolist()


def ingest_csv(csv_path: str):
    df = pd.read_csv(csv_path, encoding="latin1")
    assert "id" in df.columns and "content" in df.columns, "CSV must have columns: id, content"

    chroma.delete_collection(COLLECTION_NAME)
    col = chroma.create_collection(COLLECTION_NAME)

    ids = df["id"].astype(str).tolist()
    docs = df["content"].astype(str).tolist()
    embs = embed_texts(docs)

    BATCH = 500
    for i in range(0, len(ids), BATCH):
        col.add(
            ids=ids[i:i+BATCH],
            documents=docs[i:i+BATCH],
            embeddings=embs[i:i+BATCH],
        )
    return f"✅ Ingested {len(ids)} posts into '{COLLECTION_NAME}' @ {DB_DIR}"


def retrieve_style_examples(query: str, n_results: int = 5):
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=n_results)
    docs = res.get("documents", [[]])[0]
    return [d for d in docs if d]


def generate_content(query: str, style_examples: list[str]):
    examples_block = "\n\n".join(f"- {ex}" for ex in style_examples) if style_examples else "None"
    system = f"""{SYSTEM_PROMPT}

Retrieved style examples (not to copy, only for tone):
{examples_block}
"""

    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )
    return chat.choices[0].message.content


# ====== STREAMLIT UI ======
st.set_page_config(page_title="DeepSeek RAG Generator", page_icon="📢")
st.title("📢 DeepSeek RAG LinkedIn Post Generator")

# Sidebar for ingestion
st.sidebar.header("📂 Data Management")
csv_upload = st.sidebar.file_uploader("Upload CSV (id,content)", type=["csv"])
if csv_upload and st.sidebar.button("Ingest Posts"):
    with open("uploaded_posts.csv", "wb") as f:
        f.write(csv_upload.getbuffer())
    msg = ingest_csv("uploaded_posts.csv")
    st.sidebar.success(msg)

# Main interface
query = st.text_input("👉 Enter your topic:", "")

k = st.slider("🔎 Number of style examples to retrieve", 1, 10, 5)

if st.button("Generate Post") and query.strip() != "":
    with st.spinner("✍️ Writing your post..."):
        examples = retrieve_style_examples(query, n_results=k)
        out = generate_content(query, examples)

        st.subheader("📝 Generated Post")
        st.write(out)

        if examples:
            with st.expander("🎨 Retrieved Style Examples"):
                for e in examples:
                    st.markdown(f"- {e}")


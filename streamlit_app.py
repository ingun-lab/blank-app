# streamlit_app.py

from pathlib import Path
import os, ast, numpy as np, pandas as pd, streamlit as st, gdown
from dotenv import load_dotenv
load_dotenv()  # makes OPENAI_API_KEY available to query_engine.py

from query_engine import search  # <-- no need to import load_data anymore

# ---- Page setup
st.set_page_config(page_title="🔍 Policy Memo Search", layout="wide")
st.title("🔍 Policy Navigator (Beta)")

# --- 👇 Add Expander ---
st.subheader("Search across hundreds of policy memos by topic, concept, or agency.")

with st.expander("ℹ️ About this app", expanded=True):
    st.markdown("""
**Policy Navigator (Beta)** is an experimental semantic search tool built by the **D1 Team** 
to help surface insights from our internal library of policy memos and publications.

This app uses **OpenAI’s `text-embedding-3-large` model** to process natural-language queries and return 
the most relevant memos based on meaning, not just keywords.

- 🧭 **How to use:** Type a topic, agency, or concept (e.g., *federal research capacity*, *DOE innovation programs*, *climate and health impacts*).
- ⚙️ **How it works:** Your query is embedded into vector space and compared against precomputed memo embeddings using cosine similarity.
- 🗂️ **Data size:** ~3,400 publications embedded with OpenAI and cached for fast multi-user search.
- 🌐 **Source:** FAS Salesforce Publication Repository.
- 🧠 **Maintainer:** Day One Team.

For questions or feedback, contact: [ingun@fas.org](mailto:ingun@fas.org)
""")
    
# ---- Google Drive config (make sure file is 'Anyone with the link -> Viewer')
FILE_ID = "1HUARJzdDRnNCnEriJ9iFjL4_1_TcM12W"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ---- Persistent cache (survives restarts; shared by all sessions)
DATA_DIR = Path("data_cache"); DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "policy_memos.csv"
PARQUET_PATH = DATA_DIR / "policy_memos.parquet"

def _safe_parse_embedding(x):
    if isinstance(x, list) and all(isinstance(i,(float,int)) for i in x):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list) and all(isinstance(i,(float,int)) for i in v):
                return v
        except Exception:
            pass
    return np.nan

def _download_csv_if_needed():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with st.spinner("📥 Downloading dataset from Google Drive (first time only)…"):
            gdown.download(GDRIVE_URL, str(CSV_PATH), quiet=False)
    # Guard: ensure it isn't the HTML 'confirm' page
    with open(CSV_PATH, "rb") as f:
        if b"<html" in f.read(512).lower():
            raise RuntimeError(
                "Google Drive returned HTML (confirm/permission page). "
                "Make the file public: 'Anyone with the link – Viewer'."
            )

@st.cache_resource  # server-wide; shared by all users
def load_dataset() -> pd.DataFrame:
    # Fast path: use Parquet if present
    if PARQUET_PATH.exists() and PARQUET_PATH.stat().st_size > 0:
        st.caption("✅ Using cached Parquet dataset")
        return pd.read_parquet(PARQUET_PATH)

    _download_csv_if_needed()
    st.caption("🛠️ Converting CSV → Parquet for faster loads…")

    df = pd.read_csv(CSV_PATH, encoding="utf-8", encoding_errors="ignore")
    df.rename(columns=lambda c: c.strip(), inplace=True)

    # Normalize embedding column name if needed
    if "Embedding" not in df.columns:
        for c in df.columns:
            if c.strip().lower() in {"embedding","embeddings","openai_embedding","vector","emb"}:
                df.rename(columns={c: "Embedding"}, inplace=True); break

    # Light cleanup
    for col in ["title","body_text","text","content"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Parse embedding strings if needed
    if "Embedding" in df.columns:
        df["Embedding"] = df["Embedding"].apply(_safe_parse_embedding)
        df = df.dropna(subset=["Embedding"]).reset_index(drop=True)

    # Persist Parquet for future fast loads
    df.to_parquet(PARQUET_PATH, index=False)
    return df

# Optional: a sidebar button to refresh the dataset when the Drive file changes
with st.sidebar:
    if st.button("🔄 Refresh dataset"):
        try:
            if PARQUET_PATH.exists(): PARQUET_PATH.unlink()
            if CSV_PATH.exists(): CSV_PATH.unlink()
        except Exception:
            pass
        st.cache_resource.clear()
        st.rerun()

st.sidebar.info(
    "💡 Need help with your queries? [Open the quick guide](https://docs.google.com/document/d/1XW03wkOXMojV_sJYhkTFk4gxtl0j7uZyWbaD2fJz_8c/edit?tab=t.0)",
    icon="❓"
)

# ⬇️ Use the new loader (not load_cached_data)
df = load_dataset()

# Validate required columns before searching
required = {"Embedding","title","body_text","url"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
else:
    query = st.text_input(
        "This tool uses OpenAI’s text-embedding-3-large model to process natural-language queries and return the most relevant memos."
    )
    if query:
        with st.spinner("Searching..."):
            results = search(df, query, top_k=50)
            for _, row in results.iterrows():
                preview = ". ".join(row["body_text"].split(". ")[:3]) + "..."
                st.markdown("---")
                st.markdown(f"### 🔹 {row['title']}")
                st.write(preview)
                st.caption(f"Similarity: `{row['score']:.3f}`")
                st.markdown(f"[🔗 View Full Memo]({row['url']})")

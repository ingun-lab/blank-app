import streamlit as st
from query_engine import load_data, search

st.set_page_config(page_title="🔍 Policy Memo Search", layout="wide")
st.title("🔍 Policy Navigator (Beta)")

@st.cache_resource
def load_cached_data():
    return load_data("data/preembedded_memos.csv")

df = load_cached_data()

query = st.text_input("This tool uses OpenAI’s text-embedding-3-large model to process natural-language queries and return the most relevant memos. You can enter a full question, a short phrase, or even just a keyword. Try searching by topic (e.g., “nuclear energy”), by agency (e.g., “NIH”), or by concept (e.g., “science infrastructure”). Clear, descriptive queries will yield the best results.")

if query:
    with st.spinner("Searching..."):
        results = search(df, query, top_k=50)  # ✅ Show top 50

        for idx, row in results.iterrows():
            preview = row["body_text"]
            preview = ". ".join(preview.split(". ")[:3]) + "..."  # ✅ First 3 sentences

            st.markdown(f"---")
            st.markdown(f"### 🔹 {row['title']}")  # ✅ Actual title
            st.write(preview)
            st.caption(f"Similarity: `{row['score']:.3f}`")
            st.markdown(f"[🔗 View Full Memo]({row['url']})")

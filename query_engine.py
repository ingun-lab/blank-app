import pandas as pd
import numpy as np
import openai
import ast

# If running in Streamlit, use secrets
try:
    import streamlit as st
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    pass  # Or raise an error/log if you're not running in Streamlit


# --- Cleaning Utilities ---
def clean_encoding(text):
    if isinstance(text, str):
        try:
            return text.encode('latin1').decode('utf-8')
        except UnicodeDecodeError:
            return text
    return text

def safe_parse_embedding(x):
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list) and all(isinstance(i, (float, int)) for i in parsed):
            return parsed
    except:
        pass
    return np.nan

# --- Load CSV & Preprocess ---
def load_data(csv_path):
    # Try latin1 decode → utf-8 fallback → leave as-is
    def clean_encoding(text):
        if isinstance(text, str):
            try:
                return text.encode('latin1').decode('utf-8')
            except:
                try:
                    return text.encode('cp1252').decode('utf-8')
                except:
                    return text
        return text

    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")

    df["title"] = df["title"].apply(clean_encoding)
    df["body_text"] = df["body_text"].apply(clean_encoding)
    df["Embedding"] = df["Embedding"].apply(safe_parse_embedding)

    df = df.dropna(subset=["Embedding"]).reset_index(drop=True)
    return df

# --- Embedding + Similarity Functions ---
def get_query_embedding(query, model="text-embedding-3-large"):
    response = openai.embeddings.create(input=[query], model=model)
    return response.data[0].embedding

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    return np.dot(a, b) / (norm_a * norm_b)

def search(df, query, top_k=50):
    query_embedding = get_query_embedding(query)

    def compute_score(embedding):
        if isinstance(embedding, list) and len(embedding) == len(query_embedding):
            return cosine_similarity(embedding, query_embedding)
        return np.nan

    df["score"] = df["Embedding"].apply(compute_score)
    df = df.dropna(subset=["score"])
    return df.sort_values("score", ascending=False).head(top_k)


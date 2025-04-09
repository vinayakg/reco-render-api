from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import faiss
import pickle
from sentence_transformers import SentenceTransformer

app = FastAPI()

# --- Load Data on Startup ---
user_profile_df = pd.read_excel("user_profile_data.xlsx")
model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("db/faiss_index.bin")
with open("db/faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# --- Build User Profiles ---
user_tag_profiles = defaultdict(list)
user_category_profiles = defaultdict(list)

for _, row in user_profile_df.iterrows():
    user = row['User']
    if pd.notna(row['Tags']):
        user_tag_profiles[user].extend(row['Tags'].split("|"))
    if pd.notna(row['category']):
        user_category_profiles[user].append(row['category'])

user_profiles_combined = {}
for user in user_tag_profiles:
    tags = " ".join(set(user_tag_profiles[user]))
    categories = " ".join(set(user_category_profiles[user]))
    user_profiles_combined[user] = f"{tags} {categories}".strip()

# --- Pydantic Models ---
class Details(BaseModel):
    relevance: str
    narrative: str

class Offer(BaseModel):
    id: str
    type: str
    title: str
    description: str
    image: str
    details: Details
    bookmarked: bool

# --- API Endpoint ---
@app.get("/recommendations", response_model=List[Offer])
def get_recommendations(
    user_name: str = Query(..., description="User ID"),
    query: Optional[str] = Query(None, description="Custom query text (optional)"),
    top_k: int = Query(5, description="Number of results to return")
):
    user_query = query if query else user_profiles_combined.get(user_name, "")
    if not user_query:
        return []

    query_vector = model.encode([user_query]).astype("float32")
    D, I = faiss_index.search(query_vector, top_k)

    results = []
    for idx in I[0]:
        m = metadata[idx]
        results.append({
            "id": m["id"],
            "type": m["type"],
            "title": m["title"],
            "description": m["description"],
            "image": m["image"],
            "details": {
                "relevance": f"Matched for user: '{user_name}'",
                "narrative": f"Recommended based on preferences or search: '{user_query}'"
            },
            "bookmarked": False
        })

    return results

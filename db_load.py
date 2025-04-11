import faiss
import pickle
import numpy as np
import pandas as pd
# --- 1. Compute your embeddings as usual ---
from sentence_transformers import SentenceTransformer
 
model = SentenceTransformer("all-MiniLM-L6-v2")
 
 
merchant_df = pd.read_csv("./data/merchants.csv")
 
 
# Clean and Prepare Merchant Data
 
merchant_df_cleaned = merchant_df[['cdf_offer_id', 'cdf_merchant_id', 'offer_name',
 
                                   'merchant_description', 'category', 'curated_image', 'Tags']].dropna()
 
merchant_df_cleaned["text"] = (
    merchant_df_cleaned["offer_name"] + ". " +
 
    merchant_df_cleaned["merchant_description"] + ". " +
 
    merchant_df_cleaned["Tags"]
 
)
 
# Your cleaned text data for embedding
text_data = merchant_df_cleaned["text"].tolist()
 
# Generate embeddings
offer_embeddings = model.encode(text_data, show_progress_bar=True)
 
# --- 2. Create FAISS Index ---
dimension = offer_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(offer_embeddings).astype("float32"))
 
# --- 3. Prepare Metadata ---
metadata = []
for _, row in merchant_df_cleaned.iterrows():
    metadata.append({
        "id": str(row["cdf_offer_id"]),
        "user_id": row["cdf_merchant_id"],
        "type": row["category"],
        "title": row["offer_name"],
        "description": row["merchant_description"],
        "image": row["curated_image"],
        "tags": row["Tags"]
    })
 
# --- 4. Save FAISS Index to File ---
faiss.write_index(faiss_index, "db/faiss_index.bin")
# --- 5. Save Metadata to File ---
with open("db/faiss_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
 
print("✅ FAISS index and metadata saved locally.")

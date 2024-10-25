from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define file path for saving/loading embeddings
EMBEDDINGS_FILE = 'embeddings.npy'
CONTENTS_FILE = 'contexts.npy'
API_KEY = "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"
api_key_header = APIKeyHeader(name="X-API-Key")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str  # Updated type hint to 'str'
    csv_file_path: str = "/home/rajneel18/Documents/mumbai-hacks/uploads/mutual_funds_data.csv"  # Path to the CSV file

def embed_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1)

def cosine_similarity(embeddings1, embeddings2):
    return torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df

def save_embeddings(embeddings, contexts):
    np.save(EMBEDDINGS_FILE, embeddings.numpy())
    np.save(CONTENTS_FILE, contexts)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CONTENTS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        contexts = np.load(CONTENTS_FILE, allow_pickle=True).tolist()
        return torch.tensor(embeddings), contexts
    return None, None

def get_relevant_contexts(question, df, top_k=5):
    question_embedding = embed_text(question)

    # Load existing embeddings if available
    all_context_embeddings, all_contexts = load_embeddings()
    if all_context_embeddings is None or all_contexts is None:
        # Combine relevant columns into a single string for each fund
        all_contexts = df.apply(lambda row: f"{row['scheme_name']}, SIP: {row['min_sip']}, Lumpsum: {row['min_lumpsum']}, "
                                            f"Expense Ratio: {row['expense_ratio']}%, Fund Size: {row['fund_size_cr']} Cr, "
                                            f"Fund Age: {row['fund_age_yr']} years, Manager: {row['fund_manager']}, "
                                            f"Category: {row['category']} - {row['sub_category']}, "
                                            f"Risk Level: {row['risk_level']}, Returns 1Y/3Y/5Y: {row['returns_1yr']}%/"
                                            f"{row['returns_3yr']}%/{row['returns_5yr']}%", axis=1).tolist()

        # Embed all contexts
        all_context_embeddings = [embed_text(context) for context in tqdm(all_contexts, desc="Embedding contexts")]
        all_context_embeddings = torch.vstack(all_context_embeddings)

        # Save embeddings for future use
        save_embeddings(all_context_embeddings, all_contexts)
    else:
        print("Loaded existing embeddings.")

    # Calculate similarities
    similarities = cosine_similarity(question_embedding, all_context_embeddings)
    top_results = similarities.topk(k=top_k)
    
    indices = top_results.indices.squeeze().tolist()
    relevant_contexts = [all_contexts[i] for i in indices]
    return relevant_contexts

def generate_answer_with_gemini(question, relevant_contexts):
    combined_context = " ".join(relevant_contexts)
    answer = generate_answer(question, combined_context)
    return answer

def generate_answer(question, context):
    return f"Answer to '{question}' based on context: {context}"

def answer_question(question, csv_file_path):
    df = load_csv_file(csv_file_path)
    relevant_contexts = get_relevant_contexts(question, df, top_k=5)
    if relevant_contexts:
        final_answer = generate_answer_with_gemini(question, relevant_contexts)
    else:
        final_answer = "Not enough relevant data found to provide an answer."
    return final_answer

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

    try:
        answer = answer_question(request.question, request.csv_file_path)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

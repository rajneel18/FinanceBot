# question_answering.py
from groq import Groq
from sentence import answer_question  # Ensure 'sentence.py' is in the same directory or adjust the import path

def call_groq_api(question, context):
    client = Groq()
    messages = [
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext: {context}\n\nAnswer:"
        }
    ]
    
    # Create a chat completion using the specified Groq model
    completion = client.chat.completions.create(
        model="llama3-8b-8192",   # Use the specified model
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )
    
    # Collect and combine chunks from the response stream to form the full answer
    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""
    
    return answer.strip() if answer else "No answer could be generated from the model."

def ask_question_with_groq(question, csv_file_path):
    # Get relevant contexts from the CSV data
    relevant_contexts = answer_question(question, csv_file_path)
    
    if relevant_contexts:
        # Join the relevant contexts to form a single input string
        combined_context = " ".join(relevant_contexts)
        # Generate an answer with the Groq API
        answer = call_groq_api(question, combined_context)
    else:
        answer = "Not enough relevant data found to provide an answer."
    
    return answer

if __name__ == "__main__":
    csv_file_path = '/home/rajneel18/Documents/mumbai-hacks/uploads/mutual_funds_data.csv'
    question = "give me more info about 1. Baroda BNPP Paribas Money Market Fund 2. Baroda BNPP Paribas Multi Cap Fund"
    
    # Get the answer to the question using the Groq model
    answer = ask_question_with_groq(question, csv_file_path)
    print("Answer:", answer)

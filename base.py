import spacy
from transformers import pipeline

# Load spaCy for document processing
nlp = spacy.load("en_core_web_sm")

# Define the document you want the chatbot to analyze
document =dialogue.txt

# Preprocess the document
document_doc = nlp(document)

# Create a question-answering pipeline using BERT
question_answering = pipeline("question-answering", model="bert-base-cased", tokenizer="bert-base-cased")

# Main chat loop
print("Chatbot: Hello! I can answer questions based on the document you provide. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        print("Chatbot: Goodbye! Have a great day!")
        break
    
    # Use the question-answering pipeline to find the answer within the document
    answer = question_answering(question=user_input, context=document)
    
    print(f"Chatbot: {answer['answer']}")

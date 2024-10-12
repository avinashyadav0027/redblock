import re
import PyPDF2
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoModel
from transformers import BertTokenizer




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained("bert-base-uncased")


def text_processing(text):

    text = text.replace("<EOS>", "").replace("<pad>", "").strip()

    lines = text.splitlines()
    processed_text = []
    current_sentence = []
    
    for line in lines:
        line = line.strip()
        
        # Skip lines that look like page numbers or references (e.g., "[30] Ofir Press")
        if re.match(r'\[\d+\]', line) or re.match(r'^\d+$', line):
            continue
        
        # Handle paragraph breaks
        if not line:
            if current_sentence:
                processed_text.append(" ".join(current_sentence))
                current_sentence = []
            processed_text.append("")  # Add an empty line for paragraph break
        else:
            # Handle hyphenated words at line breaks
            if line.endswith('-'):
                current_sentence.append(line[:-1])  # Remove the hyphen and continue the word
            elif line.endswith(('.', '!', '?')):
                current_sentence.append(line)
                processed_text.append(" ".join(current_sentence))
                current_sentence = []
            else:
                current_sentence.append(line)

    # Append any remaining sentence
    if current_sentence:
        processed_text.append(" ".join(current_sentence))

    # Join the lines into final text, with extra spaces removed
    final_text = "\n".join(processed_text)
    final_text = re.sub(r'\s+', ' ', final_text)  # Normalize excessive spaces
    
    return final_text

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def chunk_text_by_characters(text, max_chunk_size=512):
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]


def get_vector_embeddings(file_path):

    vector_embeddings = {}
    ind_to_chunks = {}
   
    extracted_text = text_processing(extract_text_from_pdf('attention.pdf'))
    chunks = chunk_text_by_characters(extracted_text, max_chunk_size=512)

    for num, chunk in enumerate(chunks):
        
        print(f'processing chunk number: {num} out of {len(chunks)} chunks')
        inputs = tokenizer(chunk.strip(), return_tensors='pt', padding=True, truncation=True)

        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None) 
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            model_outputs = model(input_ids, token_type_ids, attention_mask)
            model_embeddings = model_outputs.last_hidden_state  
            vector_embeddings[num] = model_embeddings
            ind_to_chunks[num] = chunk

    return (vector_embeddings, ind_to_chunks)

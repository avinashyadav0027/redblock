import torch
import torch.nn.functional as F

from transformers import AutoModel
from transformers import BertTokenizer

import requests
class NoVerifySession(requests.Session):
    def __init__(self, *args, **kwargs):
        super(NoVerifySession, self).__init__(*args, **kwargs)
        self.verify = False

requests.Session = NoVerifySession

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from ingestion import get_vector_embeddings




#to calculate similarity between vectors
def cosine_similarity(vector1, vector2):

    vector1_mean = torch.mean(vector1, dim=1)  
    vector2_mean = torch.mean(vector2, dim=1)  
    
    vector1_norm = F.normalize(vector1_mean, p=2, dim=-1)  
    vector2_norm = F.normalize(vector2_mean, p=2, dim=-1)  

    
    similarity = torch.sum(vector1_norm * vector2_norm)

    return similarity.item()  

#BERT for creating vector representations
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained("bert-base-uncased")

#Replace attention.pdf with any file path you want 
embedding_output = get_vector_embeddings('attention.pdf')

vector_embeddings = embedding_output[0]
ind_to_chunk = embedding_output[1]

prompt = str(input("please enter your prompt: "))

prompt_lines = tokenizer(prompt.strip(), return_tensors='pt', padding=True, truncation=True)

input_ids = prompt_lines['input_ids']
token_type_ids = prompt_lines.get('token_type_ids', None) 
attention_mask = prompt_lines['attention_mask']

with torch.no_grad():
    outputs = model(input_ids, token_type_ids, attention_mask)
    embeddings = outputs.last_hidden_state  
    prompt_vector = embeddings

simil_array = []

for i,vector in vector_embeddings.items():
    simil_score = cosine_similarity(vector,prompt_vector)
    simil_array.append([simil_score, i])

simil_array.sort()

top = 5

for i in range(0,top):
    curr_ind = simil_array[i][1]
    doc = ind_to_chunk[curr_ind]
 
    prompt = prompt + ' you can use the following information ' + doc


gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=gpt_tokenizer.eos_token_id)

gpt_inputs = gpt_tokenizer.encode(prompt, return_tensors="pt")

gpt_outputs = gpt_model.generate(
    gpt_inputs,
    max_length=1024,
    num_return_sequences=1,
    temperature=0.3,
    top_p=0.95        
)

generated_tokens = gpt_outputs[0][len(gpt_inputs[0]):]

generated_text = gpt_tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(generated_text)



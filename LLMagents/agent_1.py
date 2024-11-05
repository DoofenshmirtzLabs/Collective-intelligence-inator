from storage import get_context,update_storage
from sentence_transformers import SentenceTransformer,util
from transformers import AutoModelForCausalLM,AutoTokenizer
from dotenv import load_dotenv
from dotenv import dotenv_values
import torch
load_dotenv()
env=dotenv_values()
top_k=env('top_k')
model_name=env('model_name')
model=AutoModelForCausalLM(model_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
Tokenizer=AutoTokenizer(model_name)
user_query=''
query_embedding=embedding_model.encode(user_query)
def context_attention(context,query_embedding):
    shared_embeddings=embedding_model.encode(context,conevert_to_tensor=True)
    cosine_scores=util.cos_sim(query_embedding,shared_embeddings)[0]
    attention_weights=torch.nn.Softmax(cosine_scores,dim=0)
    weighted_embeddings=torch.sum(attention_weights.unsqueze(1)*shared_embeddings)
    combined_embeddings=torch.concat(query_embedding,weighted_embeddings)
    return combined_embeddings
context=get_context(user_query)
combined_embeddings=context_attention(context,query_embedding)

prompt='context:{context} and now answer the following question:{query}'
inputs=Tokenizer(prompt)
output=model(inputs.input_ids)
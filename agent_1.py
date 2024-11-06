from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, BertLMHeadModel, BertTokenizer
from dotenv import load_dotenv, dotenv_values
import torch
import torch.nn as nn
from typing import List
import os
import torch.nn.functional as F

load_dotenv()
env = dotenv_values()
top_k = int(env.get('top_k', 5))
model_name = env.get('model_name', 'bert-base-uncased')

# Main Code Execution
user_query = "explain me who energy is stored in bonds between atoms"  # Define a user query
embedding_model = SentenceTransformer(model_name)
# Simple in-memory storage for embeddings
class RAGStore:
    def __init__(self):
        self.embeddings = []
        self.texts = []
    
    def add_document(self, text: str):
        embedding = embedding_model.encode(text, convert_to_tensor=True)
        self.embeddings.append(embedding)
        self.texts.append(text)
    
    def retrieve(self, query: str, top_k: int = 5):
        if not self.embeddings:
            print("No documents in RAG store.")
            return []
        
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        stacked_embeddings = torch.stack(self.embeddings)
        
        cosine_scores = util.cos_sim(query_embedding, stacked_embeddings)[0]
        
        top_results = torch.topk(cosine_scores, k=min(top_k, len(self.embeddings)))
        
        results = []
        for idx in top_results.indices:
            results.append((self.texts[idx], self.embeddings[idx]))  # Return both text and embedding
        
        return results


    def update_store_with_output(self, model_output: str, similarity_threshold: float = 0.3):
        # Encode the model output
        output_embedding = embedding_model.encode(model_output, convert_to_tensor=True)
        
        # If no embeddings, add the output directly
        if not self.embeddings:
            self.add_document(model_output)
            print("Model output added to RAG store as the first entry.")
            return
        
        # Calculate cosine similarity with stored embeddings
        stacked_embeddings = torch.stack(self.embeddings)
        cosine_scores = util.cos_sim(output_embedding, stacked_embeddings)[0]
        
        # Check if the maximum similarity is below the threshold
        max_similarity = torch.max(cosine_scores).item()
        if max_similarity < similarity_threshold:
            # If similarity is below threshold, add model output to RAG store
            self.add_document(model_output)
            print("Model output added to RAG store as it was below the similarity threshold.")
        else:
            print("Model output not added to RAG store; it is similar to existing documents.")
rag_store=RAGStore()
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "OpenAI developed the ChatGPT model.",
    "Paris is known for the Eiffel Tower.",
    "Machine learning is a subfield of artificial intelligence.",
    "energy is stored between atoms through potential energy"
]

for doc in documents:
    rag_store.add_document(doc)

# Embedding Injection Model Definition
class EmbeddingInjectionModel(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.base_model = BertLMHeadModel.from_pretrained('bert-base-uncased', is_decoder=True)
        self.tokenizer = tokenizer
        self.embedding_dim = self.base_model.config.hidden_size
        
    def prepare_embeddings(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        if input_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {input_embeddings.shape[-1]}")
        
        # Add position embeddings
        position_ids = torch.arange(input_embeddings.size(1), device=input_embeddings.device).unsqueeze(0)
        position_embeddings = self.base_model.bert.embeddings.position_embeddings(position_ids)
        input_embeddings = input_embeddings + position_embeddings
        
        # Apply layer norm and dropout
        input_embeddings = self.base_model.bert.embeddings.LayerNorm(input_embeddings)
        input_embeddings = self.base_model.bert.embeddings.dropout(input_embeddings)
            
        return input_embeddings
    
    def generate_text(self, input_embeddings: torch.Tensor, max_length: int = 50, temperature: float = 0.7, top_p: float = 0.9) -> List[str]:
        """
        Generate text from embeddings
        """
        batch_size = input_embeddings.shape[0]
        generated_tokens = []
        
        # Initial hidden states
        hidden_states = self.forward(input_embeddings)
        
        for _ in range(max_length):
            logits = self.base_model.cls(hidden_states[:, -1, :]) / temperature
            
            # Apply top-p sampling
            sorted_probs, sorted_indices = torch.sort(torch.softmax(logits, dim=-1), descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set probabilities of filtered indices to zero
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            generated_tokens.append(next_token)
            
            next_token_embeddings = self.base_model.bert.embeddings.word_embeddings(next_token)
            input_embeddings = torch.cat([input_embeddings, next_token_embeddings], dim=1)
            
            hidden_states = self.forward(input_embeddings)
            
            if all(token.item() == self.tokenizer.sep_token_id for token in next_token.squeeze()):
                break
        
        generated_texts = [
            self.tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
            for tokens in torch.cat(generated_tokens, dim=1)
        ]
        
        return generated_texts

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones(input_embeddings.shape[:2], device=input_embeddings.device)
        hidden_states = self.prepare_embeddings(input_embeddings)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        outputs = self.base_model.bert(inputs_embeds=hidden_states, attention_mask=extended_attention_mask, return_dict=True)
        
        return outputs.last_hidden_state

# Context Attention Function
def context_attention(context_results, query_embedding, embedding_model):
    context_embeddings = [embedding for _, embedding in context_results]  # Extract only embeddings
    context_embeddings = torch.stack(context_embeddings) if isinstance(context_embeddings, list) else context_embeddings
    query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding

    attn_output = F.scaled_dot_product_attention(
        query_embedding.unsqueeze(0),
        context_embeddings.unsqueeze(0),
        context_embeddings.unsqueeze(0),
    )
    attn_output = attn_output.squeeze(0)
    
    combined_embeddings = torch.cat((query_embedding, attn_output), dim=-1)
    return combined_embeddings


# Embedding Injection Function
def embedding_injection(combined_embeddings):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    injection_model = EmbeddingInjectionModel(tokenizer)
    
    generated_texts = injection_model.generate_text(
        combined_embeddings,
        max_length=20,
        temperature=0.7,
        top_p=0.9
    )
    
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated text {i+1}:")
        print(text)
    return generated_texts

# Load Environment Variables
  # Default to 5 if 'top_k' is not set

query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)

# Get context, compute attention, and generate text
context_results = rag_store.retrieve(user_query)
combined_embeddings = context_attention(context_results, query_embedding, embedding_model)
model_outputs = embedding_injection(combined_embeddings)
rag_store.update_store_with_output(model_outputs)

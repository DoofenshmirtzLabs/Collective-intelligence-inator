import torch
import torch.nn as nn
from typing import List, Optional
from transformers import BertLMHeadModel, BertTokenizer, AutoTokenizer

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
    
    def generate_text(self, 
                     input_embeddings: torch.Tensor, 
                     max_length: int = 50,
                     temperature: float = 0.7,
                     top_p: float = 0.9) -> List[str]:
        """
        Generate text from embeddings
        """
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        generated_tokens = []
        
        # Initial hidden states
        hidden_states = self.forward(input_embeddings)
        past_key_values = None
        current_length = 0
        
        while current_length < max_length:
            # Get logits from the last hidden state
            logits = self.base_model.cls(hidden_states[:, -1:, :])
            logits = logits / temperature
            
            # Apply top-p sampling
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Sample next tokens
            for idx in range(batch_size):
                probs[idx, 0, sorted_indices[idx, 0, sorted_indices_to_remove[idx, 0]]] = 0
                probs[idx] = probs[idx] / probs[idx].sum()
            
            next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            generated_tokens.append(next_token)
            
            # Get embeddings for next token
            next_token_embeddings = self.base_model.bert.embeddings.word_embeddings(next_token)
            input_embeddings = torch.cat([input_embeddings, next_token_embeddings], dim=1)
            
            # Get next hidden states
            hidden_states = self.forward(input_embeddings)
            current_length += 1
            
            # Stop if we generate EOS token
            if all(self.tokenizer.sep_token_id in seq for seq in torch.cat(generated_tokens, dim=1)):
                break
        
        # Combine all tokens and decode
        generated_tokens = torch.cat(generated_tokens, dim=1)
        generated_texts = []
        for tokens in generated_tokens:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input_embeddings.shape
        
        # Create causal attention mask
        attention_mask = torch.ones(batch_size, seq_len, device=input_embeddings.device)
        
        # Create causal mask for decoder
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_embeddings.device)
        
        # Prepare embeddings
        hidden_states = self.prepare_embeddings(input_embeddings)
        
        # Pass through BERT layers
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        outputs = self.base_model.bert(
            inputs_embeds=hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        
        return outputs.last_hidden_state

def example_usage():
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    injection_model = EmbeddingInjectionModel(tokenizer)
    
    # Create example embeddings
    batch_size = 2
    seq_len = 5
    embedding_dim = 768  # BERT's hidden size
    
    # Generate random embeddings
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Generate text
    generated_texts = injection_model.generate_text(
        embeddings,
        max_length=20,
        temperature=0.7,
        top_p=0.9
    )
    
    # Print results
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated text {i+1}:")
        print(text)
    
    return generated_texts

if __name__ == "__main__":
    example_usage()
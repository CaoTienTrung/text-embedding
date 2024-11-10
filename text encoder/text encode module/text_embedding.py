import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict
from mask.masking import generate_padding_mask
from text_data.train import train_data
import yaml

class TextEmbedding(nn.Module):
    def __init__(self, config: Dict):
        super(TextEmbedding, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.embedding = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"])
        
        # Freeze all parameters of the pre-trained model if specified
        if config['text_embedding']['freeze']:
            for param in self.embedding.parameters():
                param.requires_grad = False

        # Additional layers for projection, activation, and dropout
        self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])
        
        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Tokenizer settings
        self.padding = config["tokenizer"]["padding"]
        self.max_length = config["tokenizer"]["max_length"]
        self.truncation = config["tokenizer"]["truncation"]
        self.return_token_type_ids = config["tokenizer"]["return_token_type_ids"]
        self.return_attention_mask = config["tokenizer"]["return_attention_mask"]

    def forward(self, texts: List[str]):
        # Tokenize input texts
        inputs = self.tokenizer(
            texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors='pt',
            return_token_type_ids=self.return_token_type_ids,
            return_attention_mask=self.return_attention_mask,
        ).to(self.device)

        # Get embeddings from the model
        features = self.embedding(**inputs).last_hidden_state
        
        # Generate padding mask
        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        
        # Project embeddings to lower dimensional space, apply activation and dropout
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        
        return out, padding_mask

texts = []
for v in train_data.values():
    texts.append(v["caption"])

# Function to load YAML configurations
def load_yaml_config(filename):
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)  # Load the YAML file
    return config

# Load the configuration from visobert_config.yaml
config = load_yaml_config('../config/visobert_config.yaml')
model = TextEmbedding(config)

model.to(model.device)

embeddings, padding_mask = model(texts)

print("Embedding shape:", embeddings.shape)
print("Padding mask shape:", padding_mask.shape)

print(embeddings)


from transformers import AutoModel, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained('uitnlp/visobert')
tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')


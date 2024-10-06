from torch.utils.data import DataLoader
from dataloader import Sarcasm
import torch

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    images = [torch.Tensor(item['image']).unsqueeze(0) for item in batch]
    captions = [item['caption'] for item in batch]
    labels = [item['label'] for item in batch]
    
    return ids, images, captions, labels

if __name__ == '__main__':
    dataset = Sarcasm('vimmsd-warmup.json')
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        num_workers=2, 
        collate_fn=collate_fn
    )
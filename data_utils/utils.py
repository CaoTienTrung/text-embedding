import torch

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    labels = [item['label'] for item in batch]
    
    return ids, images, captions, labels
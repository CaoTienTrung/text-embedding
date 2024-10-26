from data_utils.dataloader import Sarcasm
from torch.utils.data import DataLoader
from data_utils.utils import collate_fn

if __name__ == '__main__':
    dataset = Sarcasm()
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        num_workers=2, 
        collate_fn=collate_fn
    )
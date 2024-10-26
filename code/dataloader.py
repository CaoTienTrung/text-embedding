import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class Sarcasm(Dataset):
    def __init__(self, file_json):
        super().__init__()
        
        with open(file_json, 'r', encoding = 'utf-8') as file:
            data_json = json.load(file)
        
        self.__data = {}
        
        folder_images = 'C:\\Users\\admin\\Desktop\\DSC_B\\warmup-images'
        
        for i_th, (idx, value) in enumerate(data_json.items()):
            image_path = folder_images + '\\' + value['image']
            img = Image.open(image_path)
            
            self.__data[i_th] = {
                'id': idx,
                'image': np.asarray(img),
                'caption': value['caption'],
                'label': value['label']
            }
            
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, index):
        return self.__data[index]
        
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        
        for root_dir, dirs, files in os.walk(root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and len(self.images) < 5000:
                    self.images.append(os.path.join(root_dir, file))
        
        print(f"Found {len(self.images)} images in dataset")
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image 
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid image instead
            return self.__getitem__(np.random.randint(0, len(self.images)))
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.masks.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.zeros((256,256))
        mask_list = []
        for i in range(7):
            mask_path = os.path.join(self.mask_dir, self.masks[index*7+i])
            mask_list.append(np.array(Image.open(mask_path).convert("L"), dtype=np.float32))
            mask_list[i] = mask_list[i]/255
            if i != 6:
                mask += mask_list[i]*(i+1)
        

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


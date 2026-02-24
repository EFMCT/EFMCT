import os
from tifffile import imread
from torch.utils.data import Dataset

class TiffDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
        self.image_filenames.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = imread(img_path)

        if self.transform:
            image = self.transform(image)

        return image  # No labels assumed, modify if needed
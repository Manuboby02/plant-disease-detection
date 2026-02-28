import os
import torch
from torch.utils.data import Dataset
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


class PlantDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

        self.transform = Compose([
            Resize(224, 224),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])

        if image is None:
            # Skip corrupted image safely
            return self.__getitem__((index + 1) % len(self.image_paths))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[index]

        augmented = self.transform(image=image)
        image = augmented["image"]

        return image, torch.tensor(label)
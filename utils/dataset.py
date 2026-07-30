from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import pandas as pd
import torch

class ConvDataset(Dataset):
    def __init__(self, root_dir, samples=None, transform=None):
        self.root_dir = Path(root_dir)
        self.class_dirs_list = [Path(class_dir) for class_dir in self.root_dir.iterdir()]
        self.class_name_list = [class_path.name for class_path in self.class_dirs_list]
        self._check_dir_path()
        self.dict_class_label = {class_name: label for label, class_name in enumerate(self.class_name_list)}
        self.transform = transform or self._default_transform()
        self.samples = samples if samples is not None else self._create_samples_img()

    def _create_samples_img(self):
        samples = []
        for label, class_path in enumerate(self.class_dirs_list):
          for fname in sorted(class_path.iterdir()):
            if fname.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
              continue
            # проверка изображения на битость файла
            try:
              with Image.open(fname) as test_img:
                test_img.verify()
              samples.append([str(fname), label])
            except Exception:
                pass
        return samples

    def _check_dir_path(self):
      for cd in self.class_dirs_list:
        if not cd.exists() or not cd.is_dir():
          raise FileNotFoundError(f"CRITICAL: Folder not found: {cd}")

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)

        return image, label



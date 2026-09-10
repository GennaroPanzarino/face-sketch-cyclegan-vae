import os
import random
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class FaceSketchDataset(Dataset):
    def __init__(self, root_dir, max_samples=None, image_size=128):
        self.root_dir = root_dir
        self.photo_dir = os.path.join(root_dir, 'photos')
        self.sketch_dir = os.path.join(root_dir, 'sketches')
        self.max_samples = max_samples
        self.image_size = image_size

        if max_samples is not None:
            self.photos = sorted(os.listdir(self.photo_dir))[:max_samples]
            self.sketches = sorted(os.listdir(self.sketch_dir))[:max_samples]
        else:
            self.photos = sorted(os.listdir(self.photo_dir))
            self.sketches = sorted(os.listdir(self.sketch_dir))
        random.shuffle(self.sketches)

        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


    def __len__(self):
        return min(len(self.sketches), len(self.photos))

    def __getitem__(self, idx):
        photopath = os.path.join(self.photo_dir, self.photos[idx])
        sketchpatch = os.path.join(self.sketch_dir, self.sketches[idx])
        ph = self.transforms(Image.open(photopath).convert('RGB'))
        sk = self.transforms(Image.open(sketchpatch).convert('RGB'))
        return {'photo': ph, 'sketch': sk}
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class SatelliteDataset(Dataset):

    def __init__(self, ir_dir, rgb_dir):

        self.ir_dir = ir_dir
        self.rgb_dir = rgb_dir

        self.images = sorted(os.listdir(ir_dir))

        self.transform_ir = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_rgb = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),
                                 (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        file = self.images[idx]

        ir = Image.open(
            os.path.join(self.ir_dir, file)
        ).convert("L")

        rgb = Image.open(
            os.path.join(self.rgb_dir, file)
        ).convert("RGB")

        return self.transform_ir(ir), self.transform_rgb(rgb)
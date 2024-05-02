import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms

def img_loader(path):
    try:
        img = Image.open(path)
        return img
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None

def get_imgs_list(ori_dirs, ucc_dirs):
    img_list = []
    for ori_imgdir in ori_dirs:
        img_name = os.path.splitext(os.path.basename(ori_imgdir))[0]
        ucc_imgdir = os.path.join(os.path.dirname(ucc_dirs[0]), img_name + '.png')

        if ucc_imgdir in ucc_dirs:
            img_list.append((ori_imgdir, ucc_imgdir))

    return img_list

class UWCCDataset(data.Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, loader=img_loader):
        super(UWCCDataset, self).__init__()

        self.img_list = get_imgs_list(ori_dirs, ucc_dirs)
        if len(self.img_list) == 0:
            raise RuntimeError('Found 0 image pairs in given directories.')

        self.train = train
        self.loader = loader

        print(f"Found {len(self.img_list)} pairs of {'training' if train else 'testing'} images")

        self.transform = transforms.Compose([
            transforms.Resize((480, 640)),  # Resize images to a consistent size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization as needed
        ])

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i]) for i in range(len(img_paths))]

        if None in sample:
            return None

        sample = [self.transform(sample[0]), self.transform(sample[1])]

        return sample

    def __len__(self):
        return len(self.img_list)

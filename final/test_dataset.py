import os

from PIL import Image
from torch.utils.data import Dataset

from final.src.comfig import TEST_PATH
from training import get_transform


class CellTestDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.image_ids = [f[:-4] for f in os.listdir(self.image_dir)]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image, _ = self.transforms(image=image, target=None)
        return {'image': image, 'image_id': image_id}

    def __len__(self):
        return len(self.image_ids)


ds_test = CellTestDataset(TEST_PATH, transforms=get_transform(train=False))


def main():
    print(ds_test[0])


if __name__ == '__main__':
    main()

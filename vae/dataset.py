"""
OASIS brain MRI slice dataset for VAE training.
Loads raw grayscale PNG slices (no segmentation labels needed for Task 1).
"""

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class OASISSliceDataset(Dataset):
    """
    Loads raw 256x256 grayscale brain MRI slices from a keras_png_slices_* folder.
    Pixel values normalised to [0, 1] (or [-1, 1] if using tanh output — see transform).
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".png")
        )
        if transform is None:
            # Default: to tensor, values in [0, 1]. Matches sigmoid decoder output.
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        img = Image.open(img_path).convert("L")  # ensure single-channel grayscale
        img = self.transform(img)
        return img  # shape: (1, 256, 256)


def get_oasis_dataloaders(data_root, batch_size=64, num_workers=4):
    """
    data_root: path to .../OASIS/ containing keras_png_slices_{train,validate,test}
    Returns train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    train_ds = OASISSliceDataset(os.path.join(data_root, "keras_png_slices_train"))
    val_ds = OASISSliceDataset(os.path.join(data_root, "keras_png_slices_validate"))
    test_ds = OASISSliceDataset(os.path.join(data_root, "keras_png_slices_test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check when run directly on Rangpur
    train_loader, val_loader, test_loader = get_oasis_dataloaders(
        "/home/groups/comp3710/OASIS", batch_size=32
    )
    batch = next(iter(train_loader))
    print("Batch shape:", batch.shape)          # expect (32, 1, 256, 256)
    print("Value range:", batch.min().item(), batch.max().item())
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
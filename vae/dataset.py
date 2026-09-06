"""
OASIS brain MRI slice dataset for VAE training.
Loads raw grayscale PNG slices, resized to 128x128 (no segmentation labels
needed for Task 1 -- that's Task 2/UNet).
"""

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class OASISSliceDataset(Dataset):
    """
    Loads brain MRI slices from a keras_png_slices_* folder, resized to
    128x128 and normalised to [0, 1] to match the sigmoid decoder output.
    """

    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.filenames = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".png")
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # -> [0, 1] range, shape (1, H, W)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        img = Image.open(img_path).convert("L")  # ensure single-channel grayscale
        img = self.transform(img)
        return img  # shape: (1, 128, 128)


def get_oasis_dataloaders(data_root, batch_size=64, num_workers=4, image_size=128):
    """
    data_root: path to .../OASIS/ containing keras_png_slices_{train,validate,test}
    Returns train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    train_ds = OASISSliceDataset(os.path.join(data_root, "keras_png_slices_train"), image_size)
    val_ds = OASISSliceDataset(os.path.join(data_root, "keras_png_slices_validate"), image_size)
    test_ds = OASISSliceDataset(os.path.join(data_root, "keras_png_slices_test"), image_size)

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
    print("Batch shape:", batch.shape)          # expect (32, 1, 128, 128)
    print("Value range:", batch.min().item(), batch.max().item())
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
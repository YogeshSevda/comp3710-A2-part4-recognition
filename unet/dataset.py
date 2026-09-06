"""
OASIS brain MRI slice + segmentation mask dataset, for UNet training (Task 2).

Pairing rule: image filename 'case_XXX_slice_Y.nii.png' pairs with mask
filename 'seg_XXX_slice_Y.nii.png' (just swap the prefix).

Mask pixel values on disk are {0, 85, 170, 255} -- these are 4 class labels
spread across the uint8 range, NOT already 0/1/2/3. We remap them to class
indices 0-3, then one-hot encode to shape (4, H, W) as required by the brief.

IMPORTANT: masks are resized with NEAREST-NEIGHBOUR interpolation, not the
default bilinear. Bilinear would blend adjacent class values together
(e.g. average of 0 and 85 = 42.5), creating a "class" that doesn't exist.
Nearest-neighbour keeps every pixel a valid class label.
"""

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Raw pixel values found on disk -> class index (0=background typically)
MASK_VALUE_TO_CLASS = {0: 0, 85: 1, 170: 2, 255: 3}
NUM_CLASSES = 4


class OASISSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=128):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.image_filenames = sorted(
            f for f in os.listdir(image_dir) if f.endswith(".png")
        )
        # Sanity check pairing up front, so a mismatch fails loudly at
        # startup instead of silently training on wrong pairs.
        missing = []
        for fname in self.image_filenames:
            mask_name = fname.replace("case", "seg", 1)
            if not os.path.exists(os.path.join(mask_dir, mask_name)):
                missing.append(fname)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no matching mask, e.g. {missing[:3]}. "
                f"Check the image/mask directories are the correct paired sets."
            )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        fname = self.image_filenames[idx]
        mask_name = fname.replace("case", "seg", 1)

        img = Image.open(os.path.join(self.image_dir, fname)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, mask_name))  # keep raw mode

        img = TF.resize(img, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size],
                          interpolation=transforms.InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)  # (1, H, W), values in [0, 1]

        mask_arr = np.array(mask)  # (H, W), raw values {0, 85, 170, 255}
        class_map = np.zeros_like(mask_arr, dtype=np.int64)
        for raw_val, class_idx in MASK_VALUE_TO_CLASS.items():
            class_map[mask_arr == raw_val] = class_idx

        # One-hot encode: (H, W) -> (NUM_CLASSES, H, W)
        mask_onehot = torch.nn.functional.one_hot(
            torch.from_numpy(class_map), num_classes=NUM_CLASSES
        ).permute(2, 0, 1).float()

        return img_t, mask_onehot


def get_oasis_seg_dataloaders(data_root, batch_size=16, num_workers=1, image_size=128):
    """
    data_root: path to .../OASIS/ containing keras_png_slices_{train,validate,test}
    and keras_png_slices_seg_{train,validate,test}
    """
    from torch.utils.data import DataLoader

    train_ds = OASISSegDataset(
        os.path.join(data_root, "keras_png_slices_train"),
        os.path.join(data_root, "keras_png_slices_seg_train"),
        image_size,
    )
    val_ds = OASISSegDataset(
        os.path.join(data_root, "keras_png_slices_validate"),
        os.path.join(data_root, "keras_png_slices_seg_validate"),
        image_size,
    )
    test_ds = OASISSegDataset(
        os.path.join(data_root, "keras_png_slices_test"),
        os.path.join(data_root, "keras_png_slices_seg_test"),
        image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check when run directly on Rangpur
    train_loader, val_loader, test_loader = get_oasis_seg_dataloaders(
        "/home/groups/comp3710/OASIS", batch_size=8
    )
    img, mask = next(iter(train_loader))
    print("Image batch shape:", img.shape)   # expect (8, 1, 128, 128)
    print("Mask batch shape:", mask.shape)   # expect (8, 4, 128, 128)
    print("Image value range:", img.min().item(), img.max().item())
    print("Mask sums to 1 per pixel (one-hot check):",
          torch.allclose(mask.sum(dim=1), torch.ones_like(mask.sum(dim=1))))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
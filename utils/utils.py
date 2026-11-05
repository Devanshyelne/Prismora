"""Dataset, preprocessing and AdaIN helpers for PRISMORA AI."""
from pathlib import Path

from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageFolderDataset(Dataset):
    """Recursively read supported images and return RGB tensors."""
    def __init__(self, root, transform=None):
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image dataset directory not found: {self.root}")
        self.files = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not any(part.startswith(".") for part in path.relative_to(self.root).parts)
        )
        if not self.files:
            raise ValueError(f"No supported images found in: {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not read dataset image: {path}") from exc
        return self.transform(image) if self.transform else image


def get_transform(size, crop, final_size, horizontal_flip=True):
    """Resize, crop and optionally flip while keeping train inputs consistent."""
    operations = []
    if size and size > 0:
        operations.append(
            transforms.Resize(
                int(size), interpolation=InterpolationMode.BILINEAR, antialias=True
            )
        )
    if crop:
        operations.append(transforms.RandomCrop(int(final_size)))
    else:
        operations.append(
            transforms.Resize(
                (int(final_size), int(final_size)),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
    if horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def calc_mean_std(features, eps=1e-5):
    if features.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] features, got shape {tuple(features.shape)}")
    batch, channels = features.shape[:2]
    flattened = features.reshape(batch, channels, -1)
    mean = flattened.mean(dim=2).reshape(batch, channels, 1, 1)
    variance = flattened.var(dim=2, unbiased=False).reshape(batch, channels, 1, 1)
    std = (variance + eps).sqrt()
    return mean, std


def adaptive_instance_normalization(content_feat, style_feat):
    if content_feat.ndim != 4 or style_feat.ndim != 4:
        raise ValueError("AdaIN expects 4D content and style feature tensors.")
    if content_feat.shape[:2] != style_feat.shape[:2]:
        raise ValueError(
            "Content and style features must have matching batch and channel dimensions."
        )
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    return (content_feat - content_mean) / content_std * style_std + style_mean

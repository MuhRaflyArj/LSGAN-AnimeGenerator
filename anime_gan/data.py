"""Dataset download and loading utilities for anime faces images."""

import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def download_dataset(
    image_dir: Path, zip_name: Path, data_root: Path, data_url: str
) -> None:
    """Download and extract the anime faces dataset if needed."""

    data_root.mkdir(parents=True, exist_ok=True)
    if image_dir.exists() and len(list(image_dir.glob("*"))) > 0:
        return

    if not zip_name.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(data_url, zip_name)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_name, "r") as zf:
        zf.extractall(data_root)


class AnimeFacesDataset(Dataset):
    """Load anime face images into a tensor dataset on the target device."""

    def __init__(self, image_dir, device, image_size=64, n_cpu=4):
        """Read, transform, and stack all images from the given folder."""

        self.device = device
        self.image_paths = sorted([p for p in Path(image_dir).glob("*") if p.is_file()])
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        images = []
        max_workers = max(1, n_cpu // 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self._load_image, self.image_paths)
            for image in tqdm(
                results, total=len(self.image_paths), desc="Loading dataset"
            ):
                images.append(image)

        self.images = torch.stack(images).to(self.device)

    def __len__(self):
        """Return the number of loaded images."""

        return self.images.shape[0]

    def __getitem__(self, idx):
        """Return one image tensor by index."""

        return self.images[idx]

    def _load_image(self, path):
        """Load and normalize a single image file."""

        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))


def create_dataloader(image_dir, device, image_size, batch_size, n_cpu):
    """Build the dataset and dataloader used for GAN training."""

    dataset = AnimeFacesDataset(
        image_dir=image_dir,
        device=device,
        image_size=image_size,
        n_cpu=n_cpu,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
    )
    return dataset, dataloader

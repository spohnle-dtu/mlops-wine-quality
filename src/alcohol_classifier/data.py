from pathlib import Path
import typer
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split


app = typer.Typer()

class AlcDataset(Dataset):
    """Our alcohol bottle images dataset."""

    def __init__(self, processed_path: Path) -> None:
        """Initialize the dataset by loading images and labels."""
        # Load the big tensors once into memory
        self.images = torch.load(processed_path / "all_images.pt")
        self.labels = torch.load(processed_path / "all_labels.pt")
        

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)
    
    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.images[index], self.labels[index]

def preprocess(data_path: Path = Path("data")) -> None:
    """
    Reads raw images, resizes them, and saves them as PyTorch tensors.
    """
    raw_dir = Path(data_path / "raw")
    processed_dir = Path(data_path / "processed")

    # check if processed directory exists, if not create it
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing data from {raw_dir}")
    
    # Define the standardized format
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    categories = ["beer", "whiskey", "wine"]

    # Create one big tensor and add category as label
    all_images = []
    all_labels = []

    print("Creating tensors from images and labels...")
    for label, category in enumerate(categories):
        image_paths = glob.glob(str(raw_dir / category / "*.jpg"))
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_transform(image)
            all_images.append(image)
            all_labels.append(label)

    # Stack all images and labels into tensors
    all_images_tensor = torch.stack(all_images)
    all_labels_tensor = torch.tensor(all_labels)

    # Save the tensors
    torch.save(all_images_tensor, processed_dir / "all_images.pt")
    torch.save(all_labels_tensor, processed_dir / "all_labels.pt")

    print(f"Images tensor shape: {all_images_tensor.shape}") # Expecting [N, 3, 224, 224]
    print(f"Labels tensor shape: {all_labels_tensor.shape}") # Expecting [N]

def show_sample(dataset: AlcDataset, index: int = 0):
    """Visualizes a single sample from the processed dataset."""
    image_tensor, label = dataset[index]
    
    # Categories for title mapping
    categories = ["beer", "whiskey", "wine"]
    
    # Permute tensor from (C, H, W) to (H, W, C) for plotting
    image_to_show = image_tensor.permute(1, 2, 0).numpy()
    
    plt.imshow(image_to_show)
    plt.title(f"Label: {categories[label]} (Index: {index})")
    plt.axis('off')
    plt.show()

@app.command()
def preprocess_check():
    """Preprocess the raw data and show a sample image."""
    preprocess()
    dataset = AlcDataset(Path("data/processed"))
    show_sample(dataset, index=700)

# ---------------------------
@dataclass
class DataConfig:
    processed_path: Path = Path("data/processed")
    batch_size: int = 32
    val_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 0  # macOS stable


def make_dataloaders(cfg: DataConfig):
    """
    Returns: train_loader, val_loader, class_names
    """
    ds = AlcDataset(cfg.processed_path)

    n = len(ds)
    n_val = int(round(n * cfg.val_fraction))
    n_train = n - n_val

    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    pin = torch.cuda.is_available()  # pin_memory only helps on CUDA

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )

    class_names = ["beer", "whiskey", "wine"]  # must match preprocess() order
    return train_loader, val_loader, class_names

if __name__ == "__main__":
    app()

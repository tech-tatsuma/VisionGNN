import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Custom collate function to create batches containing only valid samples
def custom_collate_fn(batch):
    # Filter out None values to keep only valid samples
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if len(batch) == 0:  # If the batch is empty, return None according to torch.utils.data.DataLoader behavior
        return torch.tensor([]), torch.tensor([])
    # Apply normal collate processing
    return torch.utils.data.dataloader.default_collate(batch)

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        # Initialize the path to the directory where the dataset is stored
        self.directory = directory
        # Specify the transformations to apply to the images
        self.transform = transform
        # Define a list of target classes based on the directory names
        self.classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        # Create a dictionary to convert class names to indices
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # Create a list of all samples in the dataset
        self.samples = self._make_dataset()

    def _make_dataset(self):
        # Initialize an empty list to add samples to
        samples = []
        for target in self.classes:
            # Get the path to the directory corresponding to each class
            target_dir = os.path.join(self.directory, target)
            # Skip if the directory does not exist
            if not os.path.isdir(target_dir):
                continue
            # Traverse all file names in the directory
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    # Generate the full path for each file
                    path = os.path.join(root, fname)
                    # Add a tuple of the path and corresponding class index to the list
                    item = (path, self.class_to_idx[target])
                    samples.append(item)
        return samples

    # Return the number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = None
        try:
            # Open the image file and convert it to RGB, then to grayscale if it's RGB
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
                # Check for NaN values in the image
                if torch.isnan(image).any():
                    print(f"Warning: Image at {path} contains NaN values. Skipping it.")
                    return None
            return image, target
        except (IOError, UnidentifiedImageError) as e:
            # Skip the image if it cannot be read
            print(f"Warning: Could not read image {path}. Skipping it.")
            return None, None
    
    # Additional method to return the number of classes in the dataset
    def get_num_classes(self):
        return len(self.classes)
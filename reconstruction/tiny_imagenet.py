import os
import zipfile
import shutil
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class TinyImageNetDataset:
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    ZIP_NAME = "tiny-imagenet-200.zip"
    DIR_NAME = "tiny-imagenet-200"
    
    def __init__(self, root="data", batch_size=128, num_workers=4, image_size=224):
        self.root = root
        self.data_dir = os.path.join(root, self.DIR_NAME)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.mean = [0.485, 0.456, 0.406]  # ImageNet stats
        self.std  = [0.229, 0.224, 0.225]

        self._prepare_dataset()
        self._init_datasets()

    def _prepare_dataset(self):
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            return  # already downloaded and unpacked
        
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, self.ZIP_NAME)

        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(self.URL, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)

        os.remove(zip_path)
        self._reorganize_validation_folder()

    def _reorganize_validation_folder(self):
        print("Reorganizing validation directory...")
        val_dir = os.path.join(self.data_dir, 'val')
        images_dir = os.path.join(val_dir, 'images')
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')

        with open(annotations_file, 'r') as f:
            for line in f:
                img_name, class_name = line.strip().split('\t')[:2]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                shutil.move(os.path.join(images_dir, img_name),
                            os.path.join(class_dir, img_name))

        shutil.rmtree(images_dir)

    def _init_datasets(self):
        # Use Resize(256) + Crop(224) if image_size >= 224; else direct resize
        if self.image_size >= 224:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

        train_path = os.path.join(self.data_dir, 'train')
        val_path = os.path.join(self.data_dir, 'val')

        self.train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        self.val_dataset = datasets.ImageFolder(val_path, transform=test_transform)

    def get_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader

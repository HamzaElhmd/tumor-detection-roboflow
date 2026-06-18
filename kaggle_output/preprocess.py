import torch
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.train_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

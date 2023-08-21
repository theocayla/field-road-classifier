import logging
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Dataset():

    def __init__(self, datasetRoot, batchSize=4):

        self.trainTransform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.testTransform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.batchSize = batchSize

        trainDataset = CustomImageFolder(root=os.path.join(datasetRoot, "train"), transform=self.trainTransform)
        self.trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        logging.info(f"Trainset created : {len(self.trainDataloader)} images.")

        testDataset = CustomImageFolder(root=os.path.join(datasetRoot, "test"), transform=self.testTransform)
        self.testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
        logging.info(f"Testset created : {len(self.testDataloader)} images.")

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        originalImagePath = self.imgs[index][0]
        image, label = super().__getitem__(index)
        return image, label, os.path.basename(originalImagePath)


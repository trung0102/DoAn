"""pytorchexample: A Flower / PyTorch app."""

import csv
import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CustomCIFAR10(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data.append((row[0], row[1]))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id, label_str = self.data[idx]
        img_name = os.path.join(self.root_dir, f"{img_id}.png")
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[label_str]
        
        if self.transform:
            image = self.transform(image)
            
        return {"img": image, "label": label}

def dirichlet_split(dataset, indices, num_partitions: int, alpha: float = 0.5, seed: int = 42):
    np.random.seed(seed)
    
    labels = np.array([dataset.class_to_idx[dataset.data[i][1]] for i in indices])
    num_classes = len(dataset.classes)
    
    partition_indices = [[] for _ in range(num_partitions)]
    
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        real_idx_c = [indices[i] for i in idx_c]
        np.random.shuffle(real_idx_c)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        counts = (proportions * len(real_idx_c)).astype(int)
        
        remainder = len(real_idx_c) - counts.sum()
        for _ in range(remainder):
            counts[np.random.randint(num_partitions)] += 1
            
        start = 0
        for p in range(num_partitions):
            end = start + counts[p]
            partition_indices[p].extend(real_idx_c[start:end])
            start = end
            
    return partition_indices

def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.5):
    dataset = CustomCIFAR10(
        root_dir="d:/Research/Code/DoAn/Datasets/train",
        csv_file="d:/Research/Code/DoAn/Datasets/trainLabels.csv",
        transform=pytorch_transforms
    )

    base_indices = list(range(0, 40000))
    
    #non-iid cho tập train
    partition_indices = dirichlet_split(dataset, base_indices, num_partitions, alpha=alpha, seed=42)
    my_indices = partition_indices[partition_id]
    
    train_ds = Subset(dataset, my_indices)
    
    test_ds = Subset(dataset, list(range(40000, 50000)))
    
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    dataset = CustomCIFAR10(
        root_dir="d:/Research/Code/DoAn/Datasets/train",
        csv_file="d:/Research/Code/DoAn/Datasets/trainLabels.csv",
        transform=pytorch_transforms
    )
    
    test_subset = Subset(dataset, range(40000, 50000))
    return DataLoader(test_subset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

"""pytorchexample: A Flower / PyTorch app."""

import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.model_selection import train_test_split

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


_GLOBAL_DATASET = None
_GLOBAL_PARTITIONS = None
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

def plot_data_matrix(dataset, partition_indices, num_partitions, title):
    """
    Hàm vẽ biểu đồ ma trận (Heatmap) thể hiện phân bố dữ liệu.
    """
    num_classes = len(dataset.classes)
    
    # Khởi tạo ma trận để đếm số lượng nhãn: kích thước (num_partitions, num_classes)
    client_class_counts = np.zeros((num_partitions, num_classes), dtype=int)
    
    # Đếm số lượng của từng class cho mỗi client
    for p in range(num_partitions):
        indices = partition_indices[p]
        for idx in indices:
            label_str = dataset.data[idx][1]
            label_idx = dataset.class_to_idx[label_str]
            client_class_counts[p][label_idx] += 1
            
    # Vẽ biểu đồ Heatmap bằng Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        client_class_counts, 
        annot=True,       # Hiển thị con số bên trong từng ô
        fmt="d",          # Định dạng số nguyên (integer)
        cmap="Blues",     # Thang màu từ nhạt sang đậm (có thể đổi thành "YlGnBu", "viridis"...)
        xticklabels=dataset.classes,
        yticklabels=[f"{i}" for i in range(num_partitions)],
        cbar_kws={'label': 'Number of Samples'}
    )
    
    # Làm đẹp biểu đồ
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Client', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=90, ha='right')
    
    plt.tight_layout()
    plt.savefig("d:/Research/Code/DoAn/quickstart-pytorch/pytorchexample/Non-IID.png", dpi=300, bbox_inches='tight')
    # plt.show()


def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.5):
    
    global _GLOBAL_DATASET, _GLOBAL_PARTITIONS
    
    if _GLOBAL_DATASET is None or _GLOBAL_PARTITIONS is None:
        _GLOBAL_DATASET = CustomCIFAR10(
            root_dir="d:/Research/Code/DoAn/Datasets/train",
            csv_file="d:/Research/Code/DoAn/Datasets/trainLabels.csv",
            transform=pytorch_transforms
        )

        base_indices = list(range(0, 50000))
        
        #non-iid cho tập train
        _GLOBAL_PARTITIONS = dirichlet_split(_GLOBAL_DATASET, base_indices, num_partitions, alpha=alpha, seed=42)
        plot_data_matrix(_GLOBAL_DATASET, _GLOBAL_PARTITIONS, num_partitions, f"Non-IID Data Matrix (Dirichlet $\\alpha={alpha}$)")
        print("[INIT] Chia data Non-IID\n")

    if _GLOBAL_DATASET is None or _GLOBAL_PARTITIONS is None:
        print("ERROR data Non-IID\n")
        return

    my_indices = _GLOBAL_PARTITIONS[partition_id]
    my_labels = [_GLOBAL_DATASET.class_to_idx[_GLOBAL_DATASET.data[i][1]] for i in my_indices]
    
    label_counts = Counter(my_labels)
    
    # Tách các index có lớp >= 2 mẫu và các index có lớp chỉ có 1 mẫu
    indices_to_stratify = []
    labels_to_stratify = []
    forced_train_idx = []

    for idx, lbl in zip(my_indices, my_labels):
        if label_counts[lbl] >= 2:
            indices_to_stratify.append(idx)
            labels_to_stratify.append(lbl)
        else:
            forced_train_idx.append(idx)

    
    if len(indices_to_stratify) > 1:
        train_idx, test_idx = train_test_split(
            indices_to_stratify, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels_to_stratify
        )
        train_idx = list(train_idx) + forced_train_idx
    else:
        train_idx = list(my_indices)
        test_idx = []
    
    train_ds = Subset(_GLOBAL_DATASET, train_idx)
    test_ds = Subset(_GLOBAL_DATASET, test_idx)
    
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


# if __name__ == "__main__":

#     for i in range(10):
#         print(f"User ID: {i} Load Data......")
#         load_data(i, 10, 20, 0.1)

    # # 1. Khởi tạo dataset
    # dataset = CustomCIFAR10(
    #     root_dir="d:/Research/Code/DoAn/Datasets/train",
    #     csv_file="d:/Research/Code/DoAn/Datasets/trainLabels.csv",
    #     transform=None 
    # )

    # base_indices = list(range(0, 40000))
    # num_clients = 10 
    
    # # Hàm chia IID (bổ sung nếu bạn chưa lưu)
    # def iid_split(indices, num_partitions, seed=42):
    #     np.random.seed(seed)
    #     indices_copy = list(indices).copy()
    #     np.random.shuffle(indices_copy)
    #     return [list(part) for part in np.array_split(indices_copy, num_partitions)]
    
    # # ==========================================
    # # MA TRẬN 1: DỮ LIỆU IID
    # # ==========================================
    # iid_partitions = iid_split(base_indices, num_clients, seed=42)
    # plot_data_matrix(
    #     dataset, 
    #     iid_partitions, 
    #     num_clients, 
    #     "IID Data Matrix across 10 Clients"
    # )
    
    # # ==========================================
    # # MA TRẬN 2: DỮ LIỆU NON-IID (Dirichlet)
    # # ==========================================
    # alpha_value = 0.5
    # non_iid_partitions = dirichlet_split(
    #     dataset, 
    #     base_indices, 
    #     num_clients, 
    #     alpha=alpha_value, 
    #     seed=42
    # )
    # plot_data_matrix(
    #     dataset, 
    #     non_iid_partitions, 
    #     num_clients, 
    #     f"Non-IID Data Matrix (Dirichlet $\\alpha={alpha_value}$)"
    # )
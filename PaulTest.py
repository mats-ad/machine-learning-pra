import os
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Building the dataset
def loading_the_images(image_directory):
    full_image_path = []
    labels = []

    folders = os.listdir(image_directory)

    for folder in folders:
        if not folder.__contains__('.'):  # Exclude .DS_Store
            folder_path = os.path.join(image_directory, folder)
            images = os.listdir(folder_path)
            for image in images:
                image_path = os.path.join(folder_path, image)
                full_image_path.append(image_path)
                labels.append(folder)

    image_series = pd.Series(full_image_path, name='full_image_path')
    labels_series = pd.Series(labels, name='labels')

    dataset = pd.concat([image_series, labels_series], axis=1)

    return dataset


# Classify the images
def update_labels(dataset, column_name):
    index = {'lung_aca': 'Lung_adenocarcinoma',
             'lung_n': 'Lung_benign_tissue',
             'lung_scc': 'Lung_squamous_cell_carcinoma'}

    dataset[column_name] = dataset[column_name].replace(index)


# Loading the immages
image_directory1 = '/Users/paulwieland/Repos/MLP/lung_colon_image_set/'
image_directory2 = 'lung_image_sets'
image_directory = os.path.join(image_directory1, image_directory2)

dataset = loading_the_images(image_directory)

# Update the labels
update_labels(dataset, 'labels')

# Export the dataset
"""
csv_path = '/Users/paulwieland/Repos/MLP/machine-learning-pra/image_data.csv'
dataset.to_csv(csv_path)
"""

# Split the dataset into training and testing
# Training dataset: 80% / Testing dataset: 20% (testing, validation)
train_ds, test_ds = train_test_split(dataset, train_size=0.8, test_size=0.2,
                                     shuffle=True, random_state=42)

# Split the testing dataset into testing and validation
# Testing dataset: 50% (orig. 10%) / Validation dataset: 50% (orig. 10%)
val_ds, test_ds = train_test_split(test_ds, test_size=0.5, shuffle=True,
                                   random_state=42)

# Check if the split is correct
"""print(train_ds.count())
print(val_ds.count())
print(test_ds.count())"""


# Dataset class
class CancerDetectionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label)  # .to(device)

        return image, label  # torch.tensor(label, dtype=torch.long)


"""
# Calc mean and std
class ComputeStatsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(image_path)
        image = transforms.ToTensor()(image)
        return image


if __name__ == '__main__':
    dataset = ComputeStatsDataset(dataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for images in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(f'Mean: {mean}')  # [0.6697, 0.5341, 0.8513]
    print(f'Std: {std}')  # [0.1289, 0.1742, 0.0741]
"""

# Transform and normalize the images
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6697, 0.5341, 0.8513],
                             std=[0.1289, 0.1742, 0.0741])
        # (mean=mean.toList(), std=std.tolist()),
    ])


# Training dataset and DataLoader
train_dataset = CancerDetectionDataset(train_ds, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Validation dataset and DataLoader
val_dataset = CancerDetectionDataset(val_ds, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Testing dataset and DataLoader
test_dataset = CancerDetectionDataset(test_ds, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# CNN Model
class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            # TODO PWIE: Has to be 5 in the final version (incl. collon cancer)
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)  # Is softmax the right choice?
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.flatten(output)
        output = self.fc(output)
        output = self.fc1(output)
        output = self.fc2(output)

        return output


# Initialize Model
num_classes = 3
model = AlexNet(num_classes).to(device)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training, Validating and Testing the model
total_steps = len(train_loader)

num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch+1, num_epochs, i+1, total_steps,
                  loss.item()))

    # Validating the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

    print('Accuracy on the {} validation images: {} %'
          .format(val_ds.count(), 100 * correct / total))

# Testing the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

print('Accuracy on the {} test images: {} %'
      .format(test_ds.count(), 100 * correct / total))

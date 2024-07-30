import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from data.dataset import load_data, HeadgearDataset
from models.mobileone import mobileone
from utils.train import train
from utils.test import test

def main():
    # Specify paths to your data
    dataset = {
        'train_data': '/path/to/your/train_data',
        'valid_data': '/path/to/your/valid_data',
        'test_data': '/path/to/your/test_data'
    }

    batch_size = 32
    num_classes = 1000  #Adjust this according to your dataset
    epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = load_data(dataset['train_data'])
    valid_df = load_data(dataset['valid_data'])
    test_df = load_data(dataset['test_data'])

    lb = LabelEncoder()
    train_df['encoded_labels'] = lb.fit_transform(train_df['labels'])
    valid_df['encoded_labels'] = lb.transform(valid_df['labels'])
    test_df['encoded_labels'] = lb.transform(test_df['labels'])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = HeadgearDataset(train_df, transform=transform)
    valid_dataset = HeadgearDataset(valid_df, transform=transform)
    test_dataset = HeadgearDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = mobileone(num_classes=len(lb.classes_), inference_mode=False, variant='s0').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, valid_loader, criterion, optimizer, device, epochs=epochs)
    test(model, test_loader, device)

if __name__ == '__main__':
    main()

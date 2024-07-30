import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        train_accuracy = 100 * correct / total
        train_precision = precision_score(all_labels, all_predictions, average='weighted')
        train_recall = recall_score(all_labels, all_predictions, average='weighted')
        train_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1-score: {train_f1}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_predictions = []
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
        
        val_accuracy = 100 * val_correct / val_total
        val_precision = precision_score(val_labels, val_predictions, average='weighted')
        val_recall = recall_score(val_labels, val_predictions, average='weighted')
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')
        
        print(f"Validation Loss: {val_loss/len(valid_loader)}, Validation Accuracy: {val_accuracy}, Validation Precision: {val_precision}, Validation Recall: {val_recall}, Validation F1-score: {val_f1}")

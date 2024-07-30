import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def test(model, test_loader, device):
    model.eval()
    test_correct = 0
    test_total = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    test_precision = precision_score(test_labels, test_predictions, average='weighted')
    test_recall = recall_score(test_labels, test_predictions, average='weighted')
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1-score: {test_f1}")

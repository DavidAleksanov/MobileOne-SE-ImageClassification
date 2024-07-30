from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(labels, predictions):
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return precision, recall, f1

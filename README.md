# MobileOne-SE-ImageClassification

This repository contains code for training and testing a headgear classification model using the MobileOne architecture.

## Project Structure

```
mobileone-headgear-classification/
|-- data/
|   |-- dataset.py
|-- models/
|   |-- mobileone.py
|   |-- mobileone_block.py
|   |-- se_block.py
|-- utils/
|   |-- metrics.py
|   |-- test.py
|   |-- train.py
|-- main.py
|-- requirements.txt
|-- README.md
```

- **data/dataset.py**: Contains functions and classes for loading and preprocessing the dataset.
- **models/mobileone.py**: Defines the main MobileOne model and a function to create model instances.
- **models/mobileone_block.py**: Implements the MobileOne blocks used in the network.
- **models/se_block.py**: Implements the Squeeze-and-Excitation (SE) block used for attention mechanisms.
- **utils/metrics.py**: Functions to calculate precision, recall, and F1-score.
- **utils/test.py**: Function to evaluate the model on the test dataset.
- **utils/train.py**: Function to train the model on the training dataset and validate it on the validation dataset.
- **main.py**: Main script for running the training and testing pipeline.

## Features

- **MobileOne Architecture**: Efficient and lightweight model suitable for mobile and edge devices.
- **Custom Dataset Loading**: Supports custom datasets for headgear classification.
- **Training and Evaluation**: Scripts for training the model and evaluating its performance.
- **Metrics Calculation**: Calculates precision, recall, and F1-score to assess model performance.

## Prerequisites

- Python 3.x
- PyTorch 1.7.0 or higher
- torchvision
- scikit-learn
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/daroneeee/mobileone-headgear-classification.git
   cd mobileone-headgear-classification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Prepare the Dataset

Ensure your dataset is organized as follows:
```
dataset/
|-- train/
|   |-- class1/
|   |-- class2/
|-- valid/
|   |-- class1/
|   |-- class2/
|-- test/
|   |-- class1/
|   |-- class2/
```
Adjust the paths in `main.py` to point to your dataset directories.

### Train the Model

To train the model, run:
```bash
python main.py
```

### Test the Model

Testing is included in the `main.py` script and will be executed after training. The script evaluates the model on the test dataset and prints performance metrics.

---

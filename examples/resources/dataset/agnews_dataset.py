import torch
import numpy as np
import pandas as pd
import os
import urllib.request
import csv
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer
from appfl.misc.data import (
    Dataset,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
)

def download_agnews_data():
    """Download AG_NEWS dataset if not already present."""
    base_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
    data_dir = os.getcwd() + "/datasets/RawData/agnews"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    files = {
        'train.csv': 'train.csv',
        'test.csv': 'test.csv'
    }
    
    for filename, url_filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            url = base_url + url_filename
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
    
    return data_dir

def load_agnews_csv(filepath):
    """Load AG_NEWS CSV file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                label = int(row[0])
                title = row[1]
                description = row[2]
                text = title + " " + description
                data.append((label, text))
    return data


def get_agnews(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    """
    Return the AG_NEWS dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    :param partition_strategy: data partitioning strategy
    """
    
    # Download and load AG_NEWS dataset
    data_dir = download_agnews_data()
    
    train_data = load_agnews_csv(os.path.join(data_dir, 'train.csv'))
    test_data = load_agnews_csv(os.path.join(data_dir, 'test.csv'))
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def process_data(data):
        """Process raw AG_NEWS data into tensors"""
        texts = []
        labels = []
        
        for label, text in data:
            # Convert to 0-indexed labels
            labels.append(label - 1)
            # Tokenize text
            encoded = tokenizer(
                text, 
                max_length=128, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            texts.append(encoded["input_ids"].squeeze(0))
        
        return torch.stack(texts), torch.tensor(labels)
    
    # Process test data
    test_texts, test_labels = process_data(test_data)
    test_dataset = Dataset(test_texts.float(), test_labels)
    
    # Create a temporary dataset for partitioning
    class TempDataset(TorchDataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            label, text = self.data[idx]
            encoded = tokenizer(
                text, 
                max_length=128, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            return encoded["input_ids"].squeeze(0).float(), label - 1
    
    temp_dataset = TempDataset(train_data)
    
    # Partition the dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(temp_dataset, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(
            temp_dataset, num_clients, num_of_classes=4, **kwargs
        )
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(
            temp_dataset, num_clients, **kwargs
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")
    
    return train_datasets[client_id], test_dataset


def get_agnews_val_global():
    """
    Return the AG_NEWS validation dataset for global model evaluation.
    """
    # Download and load AG_NEWS dataset
    data_dir = download_agnews_data()
    test_data = load_agnews_csv(os.path.join(data_dir, 'test.csv'))
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process test data
    texts = []
    labels = []
    
    for label, text in test_data:
        # Convert to 0-indexed labels
        labels.append(label - 1)
        # Tokenize text
        encoded = tokenizer(
            text, 
            max_length=128, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        texts.append(encoded["input_ids"].squeeze(0))
    
    test_texts = torch.stack(texts)
    test_labels = torch.tensor(labels)
    test_dataset = Dataset(test_texts.float(), test_labels)
    
    return test_dataset 
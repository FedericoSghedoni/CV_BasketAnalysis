import numpy as np
from ultralytics import YOLO
from transformer import Transformer
from datasetCreator import loadDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch.nn as nn
import torch.optim as optim
import torch

def YoloTrainer():
    # Load a model
    model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='dataset/data.yaml', epochs=100, imgsz=640, batch=16, name='yolov8s_custom')
    model.val()
    model.export(format="onnx")

def TrasformerTrainer():
    # src_size = 1008
    tgt_size = 1
    d_model = 112 # 1008/9 
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 9
    dropout = 0.1

    transformer = Transformer(tgt_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    dataset = loadDataset(verbose=False)
    train_split = int(0.8 * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])

    train_dataloader = DataLoader(train, batch_size=32, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(100):
        epoch_losses = []
        transformer.train()
        for step, datapoint in enumerate(train_dataloader):
        # Select one video at time
            for idx, (_, _) in enumerate(datapoint):
                inputs = datapoint[idx]['emb_fea']
                # Add padding to have always the same input dimension
                padd = 112 - inputs.shape[0] # 112 stand for the src_dimension/9 the feature number
                inputs = torch.nn.functional.pad(inputs, (0,0,0,padd), mode='constant', value=0)
                labels = torch.empty((1), dtype=int)
                labels[0] = datapoint[idx]['label']
                inputs = inputs.reshape((9,112))
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = transformer(inputs)
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels.unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch % 5 == 0:
            print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
            epoch_losses = []
            # Something was strange when using this?
            # transformer.eval()
            for step, datapoint in enumerate(test_dataloader):
                # Select one video at time
                for idx, (_, _) in enumerate(datapoint):
                    inputs = datapoint[idx]['emb_fea']
                    labels = torch.empty((1), dtype=int)
                    labels[0] = datapoint[idx]['label']   
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = transformer(inputs)
                loss = criterion(outputs, labels)
                epoch_losses.append(loss.item())
            print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

    # inputs, labels = next(iter(test_dataloader))
    # inputs, labels = inputs.to(device), labels.to(device)
    # outputs = transformer(inputs)

    # print("Predicted classes", outputs.argmax(-1))
    # print("Actual classes", labels)

TrasformerTrainer()
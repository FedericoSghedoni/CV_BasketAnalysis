import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import csv

from ultralytics import YOLO
from transformer import Transformer
from datasetCreator import loadDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def report(csv_file,data):
    # Write data to the CSV file
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        existing_data = list(csvreader)
    if existing_data == []:
        existing_data = [['epoch', 'loss', 'y_pred', 'y_true']]
    existing_data.append(data)
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(existing_data)

def YoloTrainer():
    # Load a model
    model = YOLO('yolov8s.pt')  # load a pretrained model

    # Train the model
    model.train(data='dataset/data.yaml', epochs=100, imgsz=640, batch=16, name='yolov8s_custom')
    model.val()
    model.export(format="onnx")

def TrasformerTrainer():

    csv_file = 'output.csv'
    transformer = Transformer(tgt_size=1, n_feature=9,  d_model=160)

    # We use the Binary Cross Entropy since we have a 2 class problem
    criterion = nn.BCELoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    dataset = loadDataset(verbose=False)
    train_split = int(0.8 * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])

    train_dataloader = DataLoader(train, batch_size=32)
    test_dataloader = DataLoader(test, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(100):
        print(f'Current Number of epoch: {epoch}')
        epoch_losses = []
        transformer.train()
        # Select one video at time
        for datapoint in train_dataloader:
            for idx in range(datapoint['label'].shape[0]):
                # Add a dimension for the batch dimension, in this implemantion is 1
                inputs = datapoint['emb_fea'][idx].unsqueeze(0)

                # Add padding to have always the same input dimension
                # 112 stand for the src_dimension/9 the feature number
                # padd = 112 - inputs.shape[1]
                # inputs = torch.nn.functional.pad(inputs, (0,0,0,padd), mode='constant', value=0)

                labels = datapoint['label'][idx]
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = transformer(inputs)
                # print(f'Output from the model and true label: {outputs, labels}')
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
        print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))

        data = [epoch,np.mean(epoch_losses)]
        report(csv_file,data)

        if epoch % 5 == 0:
            epoch_losses = []

            transformer.eval()
            for test_datapoint in test_dataloader:
                for idx in range(test_datapoint['label'].shape[0]):
                    # Select one video at time, repeat the same steps as before 
                    inputs = test_datapoint['emb_fea'][idx].unsqueeze(0)

                    # Add padding to have always the same input dimension
                    # 112 stand for the src_dimension/9 the feature number
                    # padd = 112 - inputs.shape[1]
                    # inputs = torch.nn.functional.pad(inputs, (0,0,0,padd), mode='constant', value=0)

                    labels = test_datapoint['label'][idx]   
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = transformer(inputs)
                    loss = criterion(outputs, labels)
                    epoch_losses.append(loss.item())
            print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

    return transformer

if __name__ == '__main__':
    TrasformerTrainer()
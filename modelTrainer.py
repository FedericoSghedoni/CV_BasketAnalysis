import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import csv

from ultralytics import YOLO
from transformer import Transformer
from datasetCreator import loadDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def report(csv_file,data):
    # Write data to the CSV file
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        existing_data = list(csvreader)
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

def TransformerTrainer(pretrained_model=False, learning_rate=0.001, nhead=4, dropout_rate=0.1, num_layers=1):
    csv_train_file = 'result/train_output.csv'
    csv_test_file = 'result/test_output.csv'
    transformer = Transformer(tgt_size=1, n_feature=9,  d_model=160, nhead=nhead, dropout_rate=dropout_rate, num_layers=num_layers)

    if pretrained_model != False:
        transformer.load_state_dict(torch.load(pretrained_model))

    # We use the Binary Cross Entropy since we have a 2 class problem
    criterion = nn.BCELoss()
    # optimizer = optim.AdamW(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.SGD(transformer.parameters(), lr=learning_rate, momentum=0.9)

    dataset = loadDataset(verbose=False)
    batch_size = 32
    train_split = int(0.8 * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer.to(device)
    predictions_train = []
    true_labels_train = []
    predictions_test = []
    true_labels_test = []

    with open(csv_train_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows([['epoch','loss']])

    with open(csv_test_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows([['epoch','loss']])

    for epoch in range(101):
        epoch_losses = []
        for datapoint in train_dataloader:
            # Select one video at time
            for idx in range(datapoint['label'].shape[0]):
                inputs = datapoint['emb_fea'][idx]
                labels = datapoint['label'][idx]
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = transformer(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                if epoch == 100:
                    predictions_train.append(round(outputs.item()))
                    true_labels_train.append(labels.item())
        print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))

        data = [epoch,np.mean(epoch_losses)]
        report(csv_train_file,data)

        if epoch % 5 == 0:
            epoch_losses = []
            transformer.eval()
            with torch.no_grad():
                for test_datapoint in test_dataloader:
                    # Select one video at time, repeat the same steps as before
                    for idx in range(test_datapoint['label'].shape[0]):
                        inputs = test_datapoint['emb_fea'][idx]
                        labels = test_datapoint['label'][idx]   
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = transformer(inputs)
                        loss = criterion(outputs, labels)
                        epoch_losses.append(loss.item())
                        if epoch == 100:
                            predictions_test.append(round(outputs.item()))
                            true_labels_test.append(labels.item())
            print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

            data = [epoch,np.mean(epoch_losses)]
            report(csv_test_file,data)
    
    df = pd.read_csv('result/train_output.csv')
    sns.lineplot(data=df,x='epoch',y='loss')
    plt.savefig('result/train_result.png')

    df_test = pd.read_csv('result/test_output.csv')
    sns.lineplot(data=df_test,x='epoch',y='loss')
    plt.savefig('result/test_result.png')

    # Calculate confusion matrix
    cm_train = confusion_matrix(true_labels_train, predictions_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['canestro','fuori'])
    disp.plot()
    plt.savefig('result/cm_train.png')

    # Calculate confusion matrix
    cm_test = confusion_matrix(true_labels_test, predictions_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['canestro','fuori'])
    disp.plot()
    plt.savefig('result/cm_test.png')

    return transformer

if __name__ == '__main__':
    model_directory = 'result/model.pt'
    model = TransformerTrainer(pretrained_model=False)
    # Save the trained model to a directory
    torch.save(model.state_dict(), model_directory)

    # Load the saved model from the directory
    # transformer.load_state_dict(torch.load(model_directory))
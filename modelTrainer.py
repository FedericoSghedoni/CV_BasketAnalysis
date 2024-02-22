import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import csv

from ultralytics import YOLO
from torch.optim import lr_scheduler
from transformer import Transformer
from utils import report, result_graph
from datasetCreator import loadDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def YoloTrainer():
    # Load a model
    model = YOLO('yolov8s.pt')  # load a pretrained model

    # Train the model
    model.train(data='dataset/data.yaml', epochs=100, imgsz=640, batch=16, name='yolov8s_custom')
    model.val()
    model.export(format="onnx")

def TransformerTrainer(pretrained_model=False, learning_rate=0.001, nhead=4, dropout_rate=0.1, num_layers=1, batch_size=32):
    result_path = 'result/'
    csv_train_file = f'{result_path}train_output.csv'
    csv_test_file = f'{result_path}test_output.csv'
    transformer = Transformer(tgt_size=1, n_feature=9,  d_model=160, nhead=nhead, dropout_rate=dropout_rate, num_layers=num_layers)

    if pretrained_model != False:
        transformer.load_state_dict(torch.load(pretrained_model))

    # We use the Binary Cross Entropy since we have a 2 class problem
    criterion = nn.BCELoss()
    # optimizer = optim.AdamW(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.SGD(transformer.parameters(), lr=learning_rate, momentum=0.9)
    
    # Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)

    dataset = loadDataset(verbose=False)
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
                # Update the learning rate
                scheduler.step()
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

    result_graph(result_path, true_labels_train, true_labels_test, predictions_train, predictions_test)

    return transformer

if __name__ == '__main__':
    # Create and train your model with the current hyperparameters
    model = TransformerTrainer(batch_size=64,dropout_rate=0.5,learning_rate=0.0002,num_layers=2)
    
    # Save the trained model to a directory
    model_directory = f'result/model.pt'
    torch.save(model.state_dict(), model_directory)
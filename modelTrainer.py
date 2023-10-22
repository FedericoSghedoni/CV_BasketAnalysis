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
    tgt_size = 1
    d_model = 112
    n_feature = 9
    dropout = 0.1

    transformer = Transformer(tgt_size, n_feature,  d_model, dropout)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    dataset = loadDataset(verbose=False)
    train_split = int(0.8 * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])

    print('Extracting features from Dataset...')
    train_dataloader = DataLoader(train, batch_size=32, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(100):
        epoch_losses = []
        transformer.train()
        # Select one video at time
        for idx, datapoint in enumerate(train_dataloader):

            # Add a dimension for the batch dimension, in this implemantion is 1
            inputs = datapoint[idx]['emb_fea'].unsqueeze(0)

            # Add padding to have always the same input dimension
            # 112 stand for the src_dimension/9 the feature number
            padd = 112 - inputs.shape[1]
            inputs = torch.nn.functional.pad(inputs, (0,0,0,padd), mode='constant', value=0)
            labels = datapoint[idx]['label']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = transformer(inputs)

        print(outputs.shape, labels.shape)
        print(torch.sigmoid(outputs), labels)
        loss = criterion(torch.sigmoid(outputs), labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if epoch % 5 == 0:
            print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
            epoch_losses = []
            # Something was strange when using this?
            # transformer.eval()
            for idx, test_datapoint in enumerate(test_dataloader):

                # Select one video at time, repeat the same steps as before 
                inputs = test_datapoint[idx]['emb_fea'].unsqueeze(0)

                # Add padding to have always the same input dimension
                # 112 stand for the src_dimension/9 the feature number
                padd = 112 - inputs.shape[0]
                inputs = torch.nn.functional.pad(inputs, (0,0,0,padd), mode='constant', value=0).unsqueeze(0)
                labels = test_datapoint[idx]['label']   
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = transformer(inputs)
                loss = criterion(torch.sigmoid(outputs), labels)
                epoch_losses.append(loss.item())
            print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

    # inputs, labels = next(iter(test_dataloader))
    # inputs, labels = inputs.to(device), labels.to(device)
    # outputs = transformer(inputs)

    # print("Predicted classes", outputs.argmax(-1))
    # print("Actual classes", labels)

TrasformerTrainer()
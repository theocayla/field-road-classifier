import logging
import os

import matplotlib.pyplot as plt
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

def createModel(device, backbone="ViT"):
    '''
    Method that initializes the pretrrained model.
    Resnet50 is intialized with the weights obtained on the ImageNet classification task.
    The last layer is modified to be trained on our classification task.
    '''
    if backbone == "ResNet50":
        # Load a pretrained ResNet-50 model
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers except the final classification layer
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad = True

        # Change the final classification layer for a 2-classes task
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    else:
        # Load a pre-trained ViT model
        model = timm.create_model("vit_base_patch16_224", pretrained=True)

        # Freeze all layers except the final classification layer
        for param in model.parameters():
            param.requires_grad = False

        model.head.requires_grad = True
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 2)

    # Training loop
    model.to(device)
    logging.info(f"Device used for training : {device}.")
    return model

def trainModel(model, dataset, numEpochs, device="cpu", saveModel=False):
    '''
    Methods that trains a model and outputs the training informations
    Inputs:
        - model : pretrained Resnet model from the torchvision library
        - dataset : pytorch dataloaders for trainset and testset
        - numEpochs : number of training iterations
        - device : defines which memory should the training be performed on
        - saveModel : if True, most performing weights will be saved in a weights folder within the repository
    Output:
        - trainingResults : dict containing 
            - bestWeights : the most performing weights achieved during training
            - testAccuracies - list : history of the model accuracy on the test set during training
            - trainLosses : list : history of the model loss on the trainset during trainig
            - incorrectPredictions - list[list] : mis-predicted images
    '''
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    trainLosses = []
    testAccuracies = []
    incorrectPredictions = []
    maxAccuracy = 0

    for epoch in range(numEpochs):
        incorrectPredictions.append([])
        model.train()  # Set the model to training mode
        runningLoss = 0.0
        
        for inputs, labels, _ in dataset.trainDataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

        trainLoss = runningLoss / len(dataset.trainDataloader)
        trainLosses.append(trainLoss)

        print(f"Epoch [{epoch+1}/{numEpochs}], Train Loss: {runningLoss/len(dataset.trainDataloader):.4f}")
        
        # Testing loop
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels, imgNames in dataset.testDataloader:  # Replace 'test_loader' with your test data loader
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
                for i in range(len(predicted)):
                    if predicted[i] != labels[i]:
                        incorrectPredictions[-1].append(imgNames[i]) 

        accuracy = 100 * correct / total
        testAccuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{numEpochs}], Test Accuracy: {accuracy:.2f}%")
    
        if saveModel and accuracy > maxAccuracy:
            os.makedirs("./weights", exist_ok=True)
            savePath = "weights/model.pth"
            bestWeights = model.state_dict()
            torch.save(bestWeights, savePath)
            print(f"Model saved at {savePath}")
            maxAccuracy = accuracy
    print("Training and testing finished!")

    trainingResults = {
        "bestWeights" : bestWeights,
        "testAccuracies" : testAccuracies,
        "trainLosses" : trainLosses,
        "incorrectPredictions" : incorrectPredictions,
    }

    return trainingResults


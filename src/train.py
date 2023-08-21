# This file allows to train a mod
import torch

from dataset import Dataset
from utils import plotTrainingResults, errorCounter
from models import createModel, trainModel

classes = ["field", "road"]
datasetRoot = "./dataset"
backbone = "ResNet50" # can be "ViT" or "ResNet50"

batchSize = 50
numEpochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the dataloaders with a custom dataset class
dataset = Dataset(datasetRoot, batchSize)

# Instantiate model (pretrained resnet50 with a new fc layer)
model = createModel(device, backbone)

# Train the last layer on the field/road classification task
trainingResults = trainModel(model, dataset, numEpochs, device, saveModel=True)

# Display the train loss and test accuracy curves
plotTrainingResults(trainingResults["trainLosses"], trainingResults["testAccuracies"])

# Get the images mis-predicted
predictionErrorsDistribution = errorCounter(trainingResults["incorrectPredictions"])

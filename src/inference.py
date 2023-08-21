# This file allows to use an existing model for inference. 
# It expects a trained model to have been saved in the weights directory.

import logging
import os

import matplotlib.pyplot as plt
from PIL import Image
import timm
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms

def inference(modelPath, classLabels, testFolder, backbone="ViT"):
    '''
    Methods that allows to run the inference on an image folder.
    Inputs
        modelPath : path that points toward the model weights
        classLabels - list : class names
        testFolder - path : folder of the images that are tested
    Outputs:
        dResults - dict : dictionary containing the predicted class of each image
    '''
    if backbone == "ResNet50":
        model = resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    else:
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 2)
    try:
        model.load_state_dict(torch.load(modelPath))
    except:
        logging.error(f"Model canot be loaded. Resnet50 weights should be stored under the following path : {modelPath}.")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Inference on a folder of images
    imagePaths = [os.path.join(testFolder, filename) for filename in os.listdir(testFolder)]

    dResults = {}
    for imagePath in imagePaths:
        if os.path.splitext(imagePath)[-1].lower() in [".jpg", ".jpeg", ".png"]:
            image = Image.open(imagePath)
            inputTensor = transform(image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                try:
                    output = model(inputTensor)
                except:
                    logging.error(f"Prediction failed on the image {os.path.basename(imagePath)}")
                    break
            predictedClass = torch.argmax(output).item()
            dResults[os.path.basename(imagePath)] = classLabels[predictedClass]

            # Display the image with its predicted label
            plt.imshow(image)
            plt.title(f"Predicted: {classLabels[predictedClass]}")
            plt.show()
    
    return dResults

if __name__ == "__main__":

    classes = ["field", "road"]
    modelPath = "./weights/model.pth"
    testImages = "./test_images"
    backbone = "ResNet50" # can be "ViT" or "ResNet50"

    dResults = inference(modelPath, classes, testImages)

    print(dResults)
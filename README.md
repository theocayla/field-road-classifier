# Field/Road Classifier
This project aims at training a classifier that predicts wether an image is a picture of a field, or a road.

## Installation
To use this repository, please install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Data

The data used for training is stored in the dataset directory. It has the following structure:
- dataset/
  - train/
    - field/
        - image1.jpg
        - image2.jpg
        - ...
    - road/
        - image3.jpg
        - image4.jpg
        - ...
  - test/
    - field/
        - image5.jpg
        - image6.jpg
        - ...
    - road/
        - image7.jpg
        - image8.jpg
        - ...

## Run the scripts

To train a model, the train.py file must be executed. Training parameters can be modified: model backbone, batchSize and number of epochs.
To run the inference with a trained model, the inference.py file must be used. The path to the model's weights and the test images folder must be specified manually.
Trained weights are available in the weights directory.

## Trained models

Trained models can be downloaded with the following links:
ViT : https://drive.google.com/file/d/1uY-hZ6BjZL8bG0V2WGw7K7wE3c3zEU4x/view?usp=drive_link
Resnet50 : https://drive.google.com/file/d/1-06PrXlUpQo1q9bcxKVvBue1ZpLCGnIs/view?usp=drive_link
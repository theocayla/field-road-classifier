from collections import Counter

import matplotlib.pyplot as plt

def displayImages(dataLoader, classes, batchSize):
    '''
    Method that allows to displays image contained in a dataloader
    '''
    data_iter = iter(dataLoader)

    while True:
        try:
            images, labels = next(data_iter)
        except StopIteration:
            break
        
        for i in range(batchSize):
            
            image = images[i].permute(1, 2, 0)  # Change tensor layout from CxHxW to HxWxC
            print(image.shape)
            label = labels[i].item()
            
            plt.imshow(image)
            plt.title(f"Label: {classes[label]}")
            plt.axis('off')
            plt.show()

def plotTrainingResults(trainLosses, testAccuracies):
    '''
    Plot the train loss and test accuracy curves
    '''
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(testAccuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def errorCounter(incorrectPredictions):
  '''
  Returns a dict containing the number of mispredictions for each image
  Input :
    incorrectPredictions - list[list] : the list of mispredicted images for each epoch
  '''
  lErrors = []
  for epochErrors in incorrectPredictions:
    lErrors += epochErrors
  return Counter(lErrors)
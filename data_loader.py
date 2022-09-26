# import the necessary packages
from PIL import Image
import numpy as np
from numpy import asarray
from sklearn.model_selection import train_test_split
import os


# change directory of dataset based on computer
DIRECTORY = r"C:\Users\user\Desktop\AUB_Fall_2022\fyp\Smart-door-access-through-face-mask-detection-and-facial-recognition\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        # load the image
        image = Image.open(img_path)

        # convert image to numpy array
        image_data = asarray(image)
        data.append(image_data)

        # perform one-hot encoding on the labels
        if category == "with_mask":
            labels.append(0)
        else:
            labels.append(1)


data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

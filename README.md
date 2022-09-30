# Smart door access through face mask detection and facial recognition

### We want to create a fully functional bushiness ready project that is capable of controlling an entrance of students to AUB based on face detection and vaccination records in a data base

## To Run:

#### clone repository

```
git clone https://github.com/TahaKoleilat/Smart-door-access-through-face-mask-detection-and-facial-recognition.git
```

#### install requirements

```
pip install -r requirements.txt
```

#### run image loader

This file will load the images and labels into numpy arrays ready to input into our model

#### Change the directory to the one in that contains the dataset folder

```
python data_loader.py
```

#### run video demo using the model trained on the other branch

```
python detect_mask_video.py
```

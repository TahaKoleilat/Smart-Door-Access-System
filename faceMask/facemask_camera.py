import cv2
from keras.utils import img_to_array
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import imutils

import os
from datetime import datetime
import json


number_of_visitors = 0

with open("visitors.json", "w") as outfile:
    outfile.write("")


def addItemToJson(mask, number_of_visitors):
    # Data to be written
    dictionary = {
        "person": "visitor_%s" % number_of_visitors,
        "Mask": mask,
        "FaceMatch": "not found",
        "Period": datetime.now().isoformat()
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open("visitors.json", "a") as outfile:
        if (number_of_visitors == 1):
            outfile.write('{\"people\":[')
            outfile.write(json_object)
            outfile.write(",")
        else:
            outfile.write(json_object)
            outfile.write(",")


net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
model = load_model("model_weights_mobilenet_updated.h5")

video_capture = cv2.VideoCapture(0)

# getting the timestamp before entering the loop
# convert time string to datetime
t1 = datetime.now()

while True:
    try:

        # Capture frame-by-frame
        success, frame = video_capture.read()
        frame = imutils.resize(frame, width=800)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if (success):
                face_frame = frame[startY:endY+1, startX:endX+1]
                face_frame = cv2.resize(face_frame, (224, 224))
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_frame = img_to_array(face_frame)
                face_frame = np.expand_dims(face_frame, axis=0)
                face_frame = preprocess_input(face_frame)

                pred = model.predict(face_frame)
                text = str("Mask " + str(np.squeeze(pred)*100)
                           ) if pred >= 0.5 else str("No Mask " + str((1-np.squeeze(pred))*100))
                t2 = datetime.now()
                delta = t2 - t1
                if (delta.total_seconds() > 10):
                    number_of_visitors = number_of_visitors + 1
                    addItemToJson(text, number_of_visitors)
                    t1 = t2
                color = (0, 255, 0) if text[:4] == "Mask" else (0, 0, 255)
                label = "{}".format(text)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              color, 2)
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    except Exception as e:
        print(str(e))

        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        with open("visitors.json", 'rb+') as outfile:
            outfile.seek(-1, os.SEEK_END)
            outfile.truncate()
        with open("visitors.json", "a") as outfile:
            outfile.write("]}")
        break
# do a bit of cleanup
cv2.destroyAllWindows()
video_capture.stop()

from keras.models import model_from_json
import numpy as np
import cv2


class MaskDetectionModel(object):

    MASK_STATE = ["MASK", "NO_MASK"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)

    def predict_mask(self, img):
        self.preds = self.loaded_model.predict(img)
        return MaskDetectionModel.MASK_STATE[np.argmax(self.preds)]


facec = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = MaskDetectionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        ret, fr = self.video.read()
        # gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (128, 128), interpolation=cv2.INTER_AREA)

            pred = model.predict_mask(
                roi[np.newaxis, :, :, np.newaxis].resize(128, 128, 3))

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Capture frame-by-frame

        if ret == True:
            cv2.imshow('Frame', fr)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()


def main():
    p1 = VideoCamera()
    while True:
        p1.get_frame()


if __name__ == "__main__":
    main()

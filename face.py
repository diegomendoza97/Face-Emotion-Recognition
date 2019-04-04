from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
# detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
# emotion_model_path = 'models/model.106-0.65.hdf5'

detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'model.106-0.65.hdf5'


# hyper-parameters for bounding boxes shape
# loading models
detecter = cv2.CascadeClassifier(detection_model_path)
classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

img =  cv2.imread('disgust.jpg')

img = imutils.resize(img, width=400)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detecter.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
preds = 0
canvas = np.zeros((250, 300, 3), dtype="uint8")
if len(faces) > 0:
    faces = sorted(faces, reverse=True,
                   key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces

    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

cv2.imshow('Happy', img)
cv2.waitKey(0)
emotion, prob = enumerate(zip((EMOTIONS, preds)))
# label = f'EMOTION: {label}, with probability of {prob * 100}'
print(label)

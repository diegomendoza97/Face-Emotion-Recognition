from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# starting video streaming
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    # reading the frame
    frame = imutils.resize(frame, width=400, height=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    preds = 0
    canvas = np.zeros((frame.shape[0], frame.shape[1], 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)

            p1 = (0, (i * 30) + 5)
            p2 = (w, (i * 30) + 25)

            cv2.rectangle(canvas, p1, p2, (136,71,31), -1)
            # cv2.rectangle(canvas, p1, p2, (0, 0, 255), -1)
            cv2.putText(canvas, text, (15, (i * 30) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)
            numpy_horizontal = np.hstack((frameClone, canvas))
            cv2.imshow('Face Emotion Classifier', numpy_horizontal)
    else:

        # text = "{}: {:.2f}%".format(emotion, prob * 100)
        # w = int(prob * 300)


        w = frameClone.shape[0] / 2
        h = frameClone.shape[1] / 2

        p1 = (0, (1 * 30) + 5)
        p2 = (w, (1 * 30) + 25)
        # cv2.rectangle(canvas, p1, p2, (0, 0, 255), -1)
        cv2.putText(canvas, "0 %", (15, (1 * 30) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
        cv2.putText(frameClone, "No Face Available", (15,  20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (128, 50, 200), 2)

        numpy_horizontal = np.hstack((frameClone, canvas))
        cv2.imshow('Face Emotion Classifier', numpy_horizontal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
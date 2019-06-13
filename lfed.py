
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from trainer import Trainer
from keras.models import load_model


class LFED:

    def __init__(self, train=False):
        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Load Models
        self.faceDetection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = load_model('models/model.54-0.64.hdf5', compile=False)

        # Emotions
        self.emotionsList = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

        if train:
            self.train()

    def train(self):
        trainer = Trainer()
        trainer.execute()

    def start(self):
        # Start recording
        cam = cv2.VideoCapture(0)

        while True:
            # Get camera frame and create a copy in grayscale
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the face using the face detection model
            faces = self.faceDetection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)

            # Draw rectangle if the face was detected
            if len(faces) > 0:
                # Get the position and width, height of the detected face
                (faceX, faceY, faceWidth, faceHeight) = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

                # Extract ROI of the face,
                roi = gray[faceY:faceY + faceHeight, faceX:faceX + faceWidth]

                # Resize to 48x48 pixels
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Get all predictions
                # predictions = self.model.predict(roi[np.newaxis, :, :, np.newaxis])[0]
                predictions = self.model.predict(roi)[0]

                # Get the emotion with the highest probability
                label = self.emotionsList[predictions.argmax()]

                # Draw probabilities and rectangle on face detection
                for (i, (emotion, prob)) in enumerate(zip(self.emotionsList, predictions)):
                    text = f'{emotion} = {prob * 100:.2f}'

                    # Draw probabilities by emotion
                    cv2.putText(frame, text, (10, (i * 20) + 20), self.font, 0.45, (255, 255, 255), 2)

                    # Draw detected emotion
                    cv2.putText(frame, label, (faceX, faceY - 10), self.font, 0.45, (185, 128, 41), 2)

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (faceX, faceY), (faceX + faceWidth, faceY + faceHeight), (185, 128, 41), 2)

                    # Debugging probabilities output
                    print(f'{emotion} => {prob * 100:.2f}')

            # If face was not detected draw the following text on the frame
            else:
                cv2.putText(frame, 'No Face Detected', (20, 20), self.font, 0.45, (185, 128, 41), 2)

            # Show the frame
            cv2.imshow('LFED (Live Face Emotion Detection)', frame)

            # Press 'q' to quit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    lfed = LFED()
    lfed.start()

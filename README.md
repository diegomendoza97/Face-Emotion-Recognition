# Face-Emotion-Recognition
Face Emotion Recognition
for Artificial Vision Proyect

## Proyect
This proyects train a convolutional neural network to recognize emotions either on a real time video using the camera or either analyzing photos.

## Usage
If you'd like to try it you will have to create a model, a default one is already inside models folder, it was created using the trainer.py, this will generate a model inside the models folder, you will give a route to this model inside the videoFed.py and you should be able to utilize it with your camera.

### Dataset
The dataset used in this proyect is FER2013 a dataset utilize commonly for Facial Emotion Recognition

### Trainer.py Explanation
This code will train a Convolutional Neural Network, the code uses XCEPTION training model using Fer2013.csv, this is a well known dataset, it utilizes grayscaled, labeled images in gray scales which are of 48x48 dimmensions, we apply Convoltional Filters with different number of Filters, finally we just add patience of 50 which means if the val_loss does not  change in 50 epochs the training process is just going to terminate to keep on training without any more changes. Finally this model will be stored into a .hdf5 which will be named model.\<epoch>.\<accuracy>.hdf5

### VideoFed.py Explanation
The code utilizes a premade xml file using Haar cascades for face detection, ones the face is detected we will take the frame the camera is one and convert it to a grayscale image and resized it to 48x48 pixels and use a roi predictor to predict which emotion from the trained model has the highest prediction. After this we will take the face which we will put a square around it with the highst predicted emotion as a labeled, and to the side a bar with the live emotion prediction values.

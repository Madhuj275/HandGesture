# HandSignDetection

Step 1 - collect_images.py

This Python script is designed to collect image data from a camera, which will be used to train a classification model for 26 different classes (for all alphabets). The script uses the OpenCV library to capture images from the webcam and saves them into folders categorized by class.

How it works:

Setup the Data Directory:
First, the script checks if a directory named data exists. If not, it creates one. This directory will be used to store the images collected for each class.

Initialize Variables:
number_of_classes: Defines how many classes we are collecting data for (in this case, 26).
dataset_size: The number of images to collect per class (in this case, 100).

Start Capturing Images:
The script captures video from the webcam (cv2.VideoCapture(1)).
For each class (from 0 to 25), it creates a separate folder in the data directory.

User Prompt to Start Collecting Images:
Before collecting data for each class, the user is shown a preview from the webcam with a message "Ready? Press 'Q'!". Once the user presses 'Q', the script begins capturing images.

Image Collection Loop:
For each class, the script captures a specified number of images (100 in this case). The images are saved sequentially as 0.jpg, 1.jpg, etc., inside the corresponding class folder.

Cleanup:
After all images are collected, the script releases the camera and closes all windows.

Step 2 - create_dataset.py

This Python script extracts hand landmarks from images using the MediaPipe library, which is specifically designed for hand gesture recognition and tracking. It processes the images stored in the data directory, collects hand landmarks, and stores them in a pickle file. The data collected can then be used for training machine learning models.

How it works:

Import Libraries:
The script uses MediaPipe for hand tracking, OpenCV for image processing, and pickle for saving data. Additionally, matplotlib.pyplot is imported for potential visualization purposes.

Initialize MediaPipe Hands:
mp_hands.Hands() is initialized to detect hand landmarks in static images (i.e., not video).
The min_detection_confidence is set to 0.3, meaning the model must have at least a 30% confidence in detecting a hand for it to be recognized.

Data Collection Loop:
The script iterates over the folders in the data directory. Each folder represents a class (alphabets).
For each image in a class folder, the image is read using OpenCV and converted to RGB format since MediaPipe requires RGB input.

Hand Landmark Extraction:
The script processes the image using hands.process(). If hand landmarks are detected, the x and y coordinates of each landmark are collected.
To standardize the data, the x and y coordinates are normalized by subtracting the minimum x and y values from all landmarks in the image. This ensures the hand gesture is centered in the data regardless of its position in the image.

Store Data and Labels:
The normalized x and y values (landmark coordinates) are stored in the data list, and the corresponding label (the folder name, representing the class) is stored in the labels list.

Save Data Using Pickle:
After processing all images, the data and labels are saved to a file named data.pickle using the pickle module. This file contains all the hand landmarks data and their corresponding labels, which can be used later for training a machine learning model.

Step 3 - train_classifier.py

This Python script uses the hand landmarks data stored in data.pickle to train a machine learning model for hand gesture classification. The model used is a RandomForestClassifier from the scikit-learn library, and it will be saved for future use.

How it works:

Load the Data:
The script loads the data and labels from the data.pickle file using the pickle module. The data contains the hand landmark coordinates, and the labels represent the classes (alphabets).

Prepare Data for Training:
The data and labels are converted to NumPy arrays for easier manipulation.
The dataset is then split into training and testing sets using train_test_split().
80% of the data is used for training (x_train and y_train).
20% is used for testing (x_test and y_test).
The split is stratified to ensure that the training and testing sets maintain the same distribution of classes.

Train the Random Forest Classifier:
A RandomForestClassifier() is created and trained on the x_train and y_train data using the fit() method. This model will learn to classify hand gestures based on the hand landmarks provided in the training data.

Make Predictions and Evaluate the Model:
After training, the model is used to predict the classes of the test data (x_test) using the predict() method.
The accuracy of the model's predictions is calculated using accuracy_score(), which compares the predicted labels (y_predict) with the actual labels (y_test).
The accuracy score is then printed as a percentage of correctly classified samples.

Save the Trained Model:
Finally, the trained model is saved to a file named model.p using the pickle module. This allows the model to be loaded and reused later without needing to retrain it.

Step 4 - inference_classifier.py

This Python script performs real-time hand gesture recognition using a webcam feed. The script uses MediaPipe to detect hand landmarks in the video and a pre-trained machine learning model (loaded from model.p) to predict the hand gesture (e.g., letters A-Z). The predicted gesture is displayed on the screen with the corresponding bounding box around the hand.

How it works:

Load the Trained Model:
The script loads a pre-trained Random Forest model from the model.p file using pickle. This model is used to predict hand gestures based on the detected hand landmarks.

Initialize MediaPipe for Hand Detection:
mp_hands.Hands() is initialized for hand detection in the video stream, with a minimum detection confidence of 30%.
Drawing utilities are also initialized to visualize the hand landmarks and connections on the video feed.

Capture Video from Webcam:
The script captures video from the webcam using cv2.VideoCapture(1). The number 1 specifies which camera device to use; you can change it to 0 or another number depending on your system setup.

Process Video Frames:
For each frame captured from the webcam, the script performs the following:
Convert to RGB: The frame is converted to RGB format for use with MediaPipe.
Hand Landmark Detection: The hands.process() function detects hand landmarks in the frame.
Drawing Hand Landmarks: If hand landmarks are detected, they are drawn on the video frame using MediaPipe's drawing utilities.

Extract Hand Landmarks for Prediction:
The x and y coordinates of each detected hand landmark are extracted and normalized by subtracting the minimum x and y values. This ensures that the hand is centered for prediction.
These normalized landmark coordinates are stored in the data_aux list and used as the input features for the model.

Predict Hand Gesture:
The model predicts the gesture based on the extracted landmark coordinates (data_aux).
The predicted class is mapped to a letter using labels_dict, where each class (0-25) corresponds to a letter of the alphabet (A-Z).

Display the Prediction:
A rectangle is drawn around the detected hand on the video frame.
The predicted letter is displayed above the rectangle.

Real-Time Output:
The video feed with hand landmarks and the predicted letter is continuously displayed on the screen until the user stops the program.

Cleanup:
Once the program ends, the webcam is released, and all OpenCV windows are closed.



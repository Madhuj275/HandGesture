import os
import cv2

# Define the data directory and create it if it doesn't exist
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Constants for the number of classes and dataset size
number_of_classes = 26
dataset_size = 100

# Initialize video capture from the camera (0 for default camera, 1 for external)
cap = cv2.VideoCapture(0)  # Change to 1 if you have an external camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture. Please check your camera connection.")
    exit()

# Loop through each class
for j in range(number_of_classes):
    # Create a directory for each class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for user to get ready
    done = False
    while not done:
        ret, frame = cap.read()
        
        # Check if the frame was captured correctly
        if not ret:
            print("Error: Could not read frame. Please check your camera.")
            break
        
        # Display a message on the frame
        cv2.putText(frame, 'Ready? Press "Q" to start collecting!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Break the loop if 'Q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            done = True

    # Collect dataset images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        
        # Check if the frame was captured correctly
        if not ret:
            print("Error: Could not read frame. Please check your camera.")
            break
        
        # Display the frame
        cv2.imshow('frame', frame)
        
        # Save the frame as an image
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1
        
        # Wait for a short period to allow for frame display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Define the activities of interest
activities = ['dancing', 'cleaning', 'cooking', 'shouting', 'fighting', 'theft', 'fall', 'carrying flammable objects', 'weapons', 'baby crying']

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model by adding custom layers on top of the pre-trained model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(activities), activation='softmax'))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the video
video_path = r"C:\Users\hp\Downloads\Tere Mohalle (Lyrical) _ Besharam _ Ranbir Kapoor_ Pallavi Sharda(1080P_HD).mp4"
cap = cv2.VideoCapture(video_path)

# Initialize a list to store the frames
frames = []

# Read frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = preprocess_input(preprocessed_frame)

    # Add the preprocessed frame to the list
    frames.append(preprocessed_frame)

cap.release()

# Convert the frames to a numpy array
frames = np.array(frames)

# Make predictions on the frames
predictions = model.predict(frames)

# Calculate the average prediction for each activity
activity_scores = np.mean(predictions, axis=0)

# Get the index of the activity with the highest average prediction
dominant_activity_index = np.argmax(activity_scores)

# Get the dominant activity label
dominant_activity = activities[dominant_activity_index]

print(f"Dominant activity: {dominant_activity}")


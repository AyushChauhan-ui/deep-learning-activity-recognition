# deep-learning-activity-recognition

This code provides a simple example of using deep learning for video activity recognition. It uses a pre-trained ResNet50 model to extract features from video frames and predicts the dominant activity in the video.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- Keras

You can install the required dependencies using pip:

```
pip install opencv-python tensorflow keras
```

## Usage

1. Clone the repository or download the source code.
2. Prepare your video file and specify its path in the `video_path` variable in the code.
3. Run the script:

```
python video_activity_recognition.py
```

The script will read the frames from the video, preprocess them, and make predictions using the pre-trained model. It will then calculate the average prediction score for each activity and determine the dominant activity in the video.

## Customization

- Modify the `activities` list in the code to include the activities you want to recognize. You can add or remove activities as needed.
- Adjust the model architecture or hyperparameters according to your requirements.

## Limitations

- This is a simplified example, and the accuracy may vary depending on the complexity of the activities and the available training data.
- The code assumes that the video predominantly contains a single activity. If there are multiple activities in the video, the results may not accurately identify the dominant activity.

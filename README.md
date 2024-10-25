# Simeio

Simeio is a hand gesture recognition web application that uses computer vision and machine learning to classify hand gestures in real-time through a webcam. It allows users to interact with the system by recognizing specific gestures and displaying the classification results on a user-friendly web interface.

## Features

- **Real-time hand gesture recognition** using a pre-trained machine learning model.
- **Computer Vision Processing** powered by OpenCV and cvzone’s HandTrackingModule.
- **Web Interface** built with Flask to display the video feed and classification results.
- **Customizable Model**: Easily update or retrain the model with new data for different gesture recognition needs.

## Project Structure

```
Simeio-main/
├── Model/               # Contains the machine learning model and labels for classification
├── app.py               # Main Flask application script
├── datacollection.py    # Data collection and preprocessing script
├── trainer.py           # Model training script
├── static/              # Static files for the web interface (CSS, JS, etc.)
└── templates/           # HTML templates for rendering web pages
```

## Installation

### Prerequisites

- Python 3.x
- Flask
- OpenCV
- TensorFlow or Keras
- cvzone

### Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/Simeio.git
   cd Simeio-main
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Model Setup**:
   - Place the pre-trained model file (`keras_model.keras`) and `labels.txt` in the `Model` directory.
   - Ensure the model is compatible with Keras and supports the classification labels.

4. **Run the Application**:

   ```bash
   python app.py
   ```

5. **Access the Web App**:
   - Open a browser and go to `http://127.0.0.1:5000` to access the application.

## Usage

1. **Gesture Recognition**:
   - The webcam will activate upon starting the app.
   - Show a hand gesture in front of the camera, and the application will classify it in real-time.

2. **Customizing the Model**:
   - To train the model on new gestures, use the `trainer.py` script. Update `datacollection.py` if collecting a new dataset.

## Code Overview

- **app.py**: Main Flask application that handles video capture, hand detection, and gesture classification.
- **datacollection.py**: Manages data collection for training.
- **trainer.py**: Script for training the machine learning model.
- **Model**: Contains the pre-trained model and labels file.
- **static** and **templates**: Contain the web interface assets and HTML templates.

## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss your ideas.

## Future Enhancements

- **Mobile Support**: Develop a mobile version of the application.
- **Enhanced Error Handling**: Improve detection and error messages for better user experience.
- **Additional Gesture Sets**: Expand model to support more gestures.

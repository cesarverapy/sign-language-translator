# Simeio: Real-Time Gesture Recognition System

**Simeio** is a real-time gesture recognition system that uses machine learning to classify hand gestures captured through a webcam. This project is built using TensorFlow/Keras for the deep learning model, OpenCV for video capture, and Flask for displaying results in a web interface.

## Project Structure

```
├── .gitignore                 # Git ignore file for excluding unnecessary files
├── .env                       # Environment file for configuration (not included in version control)
├── README.md                  # Project documentation
├── app.py                     # Flask application for real-time gesture prediction
├── datacollection.py          # Script to capture and save gesture images for training
├── trainer.py                 # Model training script using the collected gesture images
├── data/
│   ├── train/                 # Folder for training data, organized by class
│   └── validation/            # Folder for validation data (optional)
├── model/
│   ├── keras_model.h5         # Saved Keras model file (optional in version control)
│   ├── keras_model.keras      # Saved Keras model file (optional in version control)
│   └── labels.txt             # Label names for each gesture class
├── static/
│   ├── css/                   # CSS files for styling the web interface
│   └── images/                # Images for the web interface
└── templates/
    ├── index.html             # Main HTML template for the web interface
    ├── pag3.html              # Additional HTML page
    └── ...                    # Other HTML templates as needed
```

## Getting Started

### Prerequisites

1. **Python 3.6+**: Ensure Python is installed.
2. **Virtual Environment (optional)**: Recommended to keep dependencies isolated.

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd simeio
    ```

2. Create and activate a virtual environment (optional):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
   - Create a `.env` file in the project root and add paths or configurations as needed:
     ```plaintext
     MODEL_PATH=./model/keras_model.keras
     LABELS_PATH=./model/labels.txt
     TRAIN_DIR=./data/train
     VAL_DIR=./data/validation
     FLASK_DEBUG=True
     ```

### Usage

#### 1. Data Collection

Run `datacollection.py` to capture and label gesture images:

```bash
python datacollection.py
```

- Press **"c"** to switch between gesture classes.
- Press **"s"** to save an image for the current class.
- Press **"q"** to quit.

Captured images will be saved in the `data/train` directory, organized by class.

#### 2. Training the Model

Run `trainer.py` to train the model using the collected images:

```bash
python trainer.py
```

This will:
- Load the images from `data/train` (and optionally `data/validation`).
- Train a model using MobileNetV2, with data augmentation for improved robustness.
- Save the trained model to `model/keras_model.keras`.

#### 3. Running the Application

Run `app.py` to start the Flask web server for real-time gesture recognition:

```bash
python app.py
```

- Open a web browser and go to `http://127.0.0.1:5000` to view the application.
- Simeio will display real-time predictions based on the gestures captured by the webcam.

### Directory Overview

- **data/**: Contains training and (optionally) validation data, organized by class.
- **model/**: Stores the trained model (`keras_model.keras`) and label names (`labels.txt`).
- **static/**: Contains CSS files and images for styling the web interface.
- **templates/**: HTML templates used by Flask for the web application interface.

### .gitignore

The `.gitignore` file is configured to exclude:
- `data/` (large datasets, optional)
- `model/*.h5` and `model/*.keras` (large model files, optional)
- `__pycache__/` and other Python cache files
- `.env` for sensitive information
- `.venv/` for virtual environment dependencies

### Environment Variables

The `.env` file is used to store configurable paths and settings:

```plaintext
MODEL_PATH=./model/keras_model.keras
LABELS_PATH=./model/labels.txt
TRAIN_DIR=./data/train
VAL_DIR=./data/validation
FLASK_DEBUG=True
```

Use the `dotenv` package to load environment variables in your Python scripts:
```python
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
LABELS_PATH = os.getenv("LABELS_PATH")
```

## Future Improvements

- **Enhance Model Accuracy**: Experiment with different architectures or fine-tuning techniques.
- **Optimize Real-Time Performance**: Consider using WebSockets for lower-latency streaming.
- **Expand Gesture Set**: Add more gestures and retrain the model for broader applications.
- **Deploy on Cloud or Edge Device**: Make Simeio accessible online or on a portable device like Raspberry Pi.

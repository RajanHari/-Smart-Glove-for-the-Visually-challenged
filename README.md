# Voice-Guided Real-Time Object Detection with Obstacle Awareness (Smart-Glove-for-the-Visually-challenged)

This project combines **real-time object detection**, **voice recognition**, and **audio feedback** to create a smart vision system that responds to voice commands, identifies requested objects in the environment, estimates their distance, and alerts for potential obstacles in the way.



## Project Structure


<details>
<summary>📁 <strong>Project Structure</strong> (click to expand)</summary>

``` bash
project-root/
│
├── src/
│ └── real_time_voice_guided_object_detection.py # Main script
│
├── scripts/
│ ├── object_detection_withdistance.py
│ ├── real_time_object_detection.py 
│ └── speechrecog1.py
│ └── speechrecog2.py
│ └── speechrecog3.py
│ └── Text_to_speech.py
│
├── models/
│ ├── MobileNetSSD_deploy.caffemodel # Pre-trained Caffe model
│ └── MobileNetSSD_deploy.prototxt.txt # Model architecture
│
├── requirements.txt
├── README.md
└── found.jpg # Reference image for focal length calibration

```

</details>



---

## Features

- 🔊 **Voice-Controlled Interface**: Interact using natural language (e.g., "Where is the chair?")
- 🎯 **Real-Time Object Detection**: Uses MobileNetSSD for detecting 20 common object classes.
- 📏 **Distance Estimation**: Calculates the distance of the detected object using a reference image.
- 🚧 **Obstacle Detection**: Alerts when obstacles block the path to the target object.
- 🕒 **Voice Time Inquiry**: Ask for the current time (e.g., "What time is it?").

---

## Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt

```


## Supported Object Classes

The system can detect the following objects:

aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep,
sofa, train, tvmonitor




## How It Works

1) Listens for your voice command (e.g., "chair").
2) Processes video from your webcam using a pre-trained deep learning model.
3) Matches detected objects to your command.
4) Estimates direction (left, center, right) and distance to the object.
5) Checks for obstacles blocking the object.
6) Speaks out the result via TTS (e.g., "move right, and an obstacle ahead")


## Usage


Make sure the found.jpg image (used for focal length calibration) is present in the root folder.

Command:

``` bash
python src/real_time_voice_guided_object_detection.py \
  --prototxt models/MobileNetSSD_deploy.prototxt.txt \
  --model models/MobileNetSSD_deploy.caffemodel

```


## Obstacle Detection Logic

If an obstacle (like a sofa, chair, or table) is in the same direction as the object, the system will warn the user via audio.







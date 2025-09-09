"""
Real-Time Object Detection with Voice Interaction and Obstacle Avoidance

USAGE:
    python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime

import cv2
import imutils
import numpy as np
import pytz
import speech_recognition as sr
from gtts import gTTS
from imutils.video import VideoStream, FPS

# -----------------------------
# Distance Calculation Helpers
# -----------------------------

def find_marker(image):
    """Convert image to grayscale, blur, and find contours to identify marker."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    contours = imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.minAreaRect(largest_contour)
    return None

def distance_to_camera(known_width, focal_length, per_width):
    """Calculate distance from camera to object using similar triangles."""
    return (known_width * focal_length) / per_width

# -----------------------------
# Constants
# -----------------------------

KNOWN_DISTANCE = 70.0  # in inches
KNOWN_WIDTH = 24.0     # in inches

# -----------------------------
# Text-to-Speech Setup
# -----------------------------

def speak(text):
    """Convert text to speech and play it."""
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("objdet.mp3")
    os.system("mpg123 objdet.mp3")

# -----------------------------
# Voice Recognition Setup
# -----------------------------

def listen_command():
    """Use microphone to listen for a voice command."""
    mic_name = "USB2.0 PC CAMERA: Audio (hw:1,0)"
    sample_rate = 48000
    chunk_size = 2048
    recognizer = sr.Recognizer()
    mic_list = sr.Microphone.list_microphone_names()

    device_id = None
    for i, name in enumerate(mic_list):
        if name == mic_name:
            device_id = i
            break

    if device_id is None:
        print("[ERROR] Microphone not found.")
        sys.exit(1)

    with sr.Microphone(device_index=device_id, sample_rate=sample_rate, chunk_size=chunk_size) as source:
        recognizer.adjust_for_ambient_noise(source)
        speak("Please speak now")
        print("[INFO] Listening...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("[INFO] You said:", command)
        return command
    except sr.UnknownValueError:
        print("[WARN] Could not understand audio")
        return None

# -----------------------------
# Main
# -----------------------------

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
    parser.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")
    parser.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum confidence threshold")
    args = vars(parser.parse_args())

    # Load the reference image for focal length calculation
    ref_image = cv2.imread("found.jpg")
    ref_marker = find_marker(ref_image)
    if not ref_marker:
        print("[ERROR] Could not find marker in reference image.")
        sys.exit(1)
    focal_length = (ref_marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    # Load model
    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # Start video stream
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    # Class labels and colors
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "glass", "bus", "car", "cat", "seat", "cow", "table",
               "dog", "horse", "motorbike", "mum", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Obstacle categories
    obstacle_classes = {"chair", "diningtable", "sofa"}

    # Get user voice input
    user_input = listen_command()
    if not user_input:
        sys.exit(0)

    # Time-based response
    if re.search("time", user_input, re.IGNORECASE):
        current_time = str(datetime.now(pytz.timezone('Asia/Kolkata')).time())
        speak(current_time)
        sys.exit()

    # Object detection
    object_found = False
    object_direction = None
    obstacle_detected = False
    obstacle_blocking = False
    distance_inches = None

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, f"{label}: {confidence * 100:.2f}%", (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                center_x = (startX + endX) / 2

                # Check if detected object matches spoken word
                if label.lower() in user_input.lower():
                    object_found = True
                    if center_x > 300:
                        object_direction = "right"
                    elif 180 < center_x <= 300:
                        object_direction = "center"
                    else:
                        object_direction = "left"

                    cv2.imwrite("found_temp.jpg", frame)
                    image = cv2.imread("found_temp.jpg")
                    marker = find_marker(image)
                    if marker:
                        distance_inches = distance_to_camera(KNOWN_WIDTH, focal_length, marker[1][0])

                # Obstacle detection
                if label in obstacle_classes and label.lower() not in user_input.lower():
                    obstacle_detected = True
                    obs_x_center = (startX + endX) / 2
                    if (object_direction == "right" and obs_x_center > 300) or \
                       (object_direction == "center" and 180 < obs_x_center <= 300) or \
                       (object_direction == "left" and obs_x_center <= 180):
                        obstacle_blocking = True

        # Voice feedback when object is found
        if object_found:
            direction_text = f"Object was found and is to the {object_direction}"
            if obstacle_blocking:
                direction_text += ". Also, obstacle ahead."
            speak(direction_text)

            print("[INFO]", direction_text)
            if distance_inches:
                print(f"[INFO] Distance: {distance_inches / 12:.2f} feet")
            break

        # Show frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Exiting...")
            break

        fps.update()

    # Cleanup
    fps.stop()
    print(f"[INFO] Elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] Approx. FPS: {fps.fps():.2f}")
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()

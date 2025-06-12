import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3  # Text-to-Speech

# Initialize Camera & Models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
engine = pyttsx3.init()

# Settings
offset = 20
imgSize = 300
word = ""  # To store the formed word

# Sign Labels
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Space"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display the detected letter box
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # 🔹 Draw the styled word box at the TOP of the screen
    cv2.rectangle(imgOutput, (50, 20), (600, 80), (0, 0, 255), cv2.FILLED)  # Red Box
    cv2.putText(imgOutput, f"Word: {word}", (60, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)  # White Text

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)

    # Append letter only when 's' is pressed
    if key == ord('s') and hands:
        if labels[index] == "Space":
            word += " "  # Add space between words
        else:
            word += labels[index]

    # Clear the formed word
    if key == ord('c'):
        word = ""

    # Text-to-Speech when 't' is pressed
    if key == ord('t'):
        engine.say(word)
        engine.runAndWait()

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

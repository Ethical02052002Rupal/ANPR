#!pip install easyocr
#!pip uninstall opencv-python-headless -y
#!pip install opencv-python-headless==4.1.2.30
#!sudo apt install tesseract-ocr
#!pip install pytesseract

import cv2
import numpy as np
import glob
#import random
from matplotlib import pyplot as plt
import easyocr
import csv
#import pytesseract
#from pytesseract import Output


# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["R"]

# Images path
images_path = glob.glob(r"Images/*.jpg")



layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
#random.shuffle(images_path)
# loop through all the images
for img_path in images_path:
  try:
    # Loading image
    img = cv2.imread(img_path)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            croppedimg = img[y:y+h, x:x+w]
            croppedimg = cv2.cvtColor(croppedimg, cv2.COLOR_BGR2GRAY)
            # Creating our sharpening filter
            #filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # Applying cv2.filter2D function on our Logo image
            #sharpen_img_2=cv2.filter2D(croppedimg,-1,filter)
            #cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
            plt.imshow(cv2.cvtColor(croppedimg, cv2.COLOR_BGR2RGB))

            #Reading the text

            #Using easyocr
            read = easyocr.Reader(['en'])
            output = read.readtext(croppedimg)
            #print(output)

            #Using pytesseract
            #text = pytesseract.image_to_string(img)
            #print(text)

            #Taking out the text
            text = output[0][-2]
            #print(text)

            #CSV file creation
            header = ['Number plate', 'Image name']
            data = [text, img_path]
            with open('text-found.csv', 'a') as f:
              writer = csv.writer(f)
              writer.writerow(header)
              writer.writerow(data)

  except Exception as e:
     #Log file creation
      f = open('log.txt', 'a')
      f.write(img_path + " " + "has the error : " + str(e) + "\n")
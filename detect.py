from centroid import CentroidTracker
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.5 #Confidence threshold
nmsThreshold = 0.6   #Non-maximum suppression threshold
inpWidth = 608       #Width of network's input image
inpHeight = 608      #Height of network's input image

ct = CentroidTracker()
(H, W) = (None, None)

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "./yolov4.cfg";
modelWeights = "./yolov4.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
def postprocess(frame, outs):
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    rects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId == 0:
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    print(width)
                    if (width<500 and height<500) :
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        rects.append([left, top, left + width, top + height])
        
    objects = ct.update(rects)
    for (objectID, centroid, ) in objects.items():
   
        text = "ID {}".format(objectID)
        
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
              
            
# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.mp4"
cap = cv.VideoCapture("./v3.mp4")
fourcc = cv.VideoWriter_fourcc(*'mp4v')
vid_writer = cv.VideoWriter(outputFile, fourcc , 20, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
results = []

while cv.waitKey(1) < 0:
    
    # get frame from the video
    
    hasFrame, frame1 = cap.read()

    frame = cv.resize(frame1, (608,608))

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break
    
    # Create a 4D blob from a frame.
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    frame2 = cv.resize(frame,(round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), interpolation = cv.INTER_AREA)

    vid_writer.write(frame2)
        
    cv.imshow(winName, frame2)
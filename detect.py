import cv2
 
img = cv2.imread('./d1.jpg')
 
net = cv2.dnn.readNetFromDarknet('./yolov4.cfg', './yolov4.weights')
 
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
 
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
 
for (classId, score, box) in zip(classIds, scores, boxes):
    if classId==0:
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 255, 0), thickness=2)
 
    # text = '%s: %.2f' % (classes[classId[0]], score)
    # cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                # color=(0, 255, 0), thickness=2)
 
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
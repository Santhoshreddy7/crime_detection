import cv2
import numpy as np
import csv
import datetime
import schedule
import time

# "C:\\Users\\jagan\\Desktop\\others\\test2.mp4"

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
# f = open('csv_file/first.csv', 'w', encoding='UTF8')

classesFile = 'data/yolo.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

modelConfiguration = '/Users/a.umashankerkumar/Downloads/yolov3_custom_traffic.cfg'
modelWeights = '/Users/a.umashankerkumar/Downloads/custom-yolov3-tiny_800.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# global frame
f = open('csv_file/first.csv', 'a', encoding='UTF8', newline="")
writer = csv.writer(f, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
frame = 0
count1 = 1
# e = datetime.datetime.now()
car_count = 0
cars = []


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    global frame
    global count1
    global car_count
    count = []
    car_count = 0

    count1 = 0
    e = datetime.datetime.now()
    frame = e.strftime("%Y-%m-%d %H:%M:%S")
    # frame += 1
    count.append(frame)
    writer.writerow(count)
    # print(frame)

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # frame = frame + 1
    # head = "cars"
    for i in indices:
        i = i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, "CAR", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if cv2.rectangle:
            # count.append("cars =")
            count1 += 1
            car_count = count1
        # if count1 == 0:
        #    break
    count.append("cars =")
    count.append(count1)
    cars.append(car_count)
    writer.writerow(count)

    '''if cv2.rectangle:
            count.append("cars =")
            count1 = 1
            count.append(count1)
            writer.writerow(count)
            # writer = csv.writer(f)

         elif cv2.rectangle == 2:
            count.append("cars =")
            count1 = 2
            count.append(count1)
            writer.writerow(count)
            print(count)
        #   print(count1)
        #else:

        #    count.append("cars =")
        #    count1 = 3
        #    count.append(count1)
        #    writer.writerow(count)
            # print(count)
            # print("cars=", count1)

'''


cars.append(count1)


def cars_avg():
    count = []
    avg = sum(cars)
    if avg >= 0:
        avg = 0
        avg = int(avg)
    count.append("avg =")
    count.append(avg)
    writer.writerow(count)


# f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%'
schedule.every(10).seconds.do(cars_avg)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    # print(len(outputs))
    # print(type(outputs[0]))
    # print(outputs[0].shape) (300, 85)
    # print(outputs[1].shape) (1200, 85)
    # print(outputs[2].shape) (4800, 85)
    # print(outputs[0][0])

    findObjects(outputs, img)
    schedule.run_pending()
    cv2.imshow('Image', img)
    cv2.waitKey(1)
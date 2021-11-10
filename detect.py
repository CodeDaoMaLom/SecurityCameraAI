import cv2
import os
import numpy as np
import dlib
from datetime import datetime
import time
import requests
import base64
import json

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

class detector:

    """Class that runs detections and objects tracking for a frame"""
    trackers = []
    writer   = None
    camName  = None
    fourcc   = None
    objColor = (150, 150, 100)
    frameNum    = 0
    fps         = 6
    timestamp   = 0
    timesMissed = 0
    mysql       = None

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    def __init__(self, camName, protocol, model):
        self.fourcc     = cv2.VideoWriter_fourcc(*'avc1')
        self.net        = cv2.dnn.readNetFromCaffe(protocol, model)
        self.camName    = camName

    def updateTrackers(self, frame):
        for trackerData in self.trackers:
            frame = self.updateTracker(frame, trackerData)
        return frame

    def updateTracker(self, frame, trackerData):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker = trackerData['tracker']
        color    = trackerData['color']
        label    = trackerData['label']
        tracker.update(rgb)
        pos     = tracker.get_position()
        startX  = int(pos.left())
        startY  = int(pos.top())
        endX    = int(pos.right())
        endY    = int(pos.bottom())
        self.updateFrame(frame, [startX, startY, endX, endY], label, color)
        return frame

    def startTracker(self, frame, box, label, color):
        tracker = dlib.correlation_tracker()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        tracker.start_track(rgb, rect)
        self.trackers.append({'tracker': tracker, 'color': color, 'label': label})

    def detect(self, frame, fps):

        if frame is None:
            return
        #Count frames for FPS calculation
        self.frameNum = self.frameNum + 1
        if self.timestamp != 0:
            self.fps = 1/(time.time() - self.timestamp)
        self.timestamp = time.time()

        #If there were no dlib correlation trackers or we just got 20 frames passed
        if not self.trackers  or self.frameNum == 5:
            # if self.frameNum < 5:
            #     return frame
            self.frameNum = 0

            detections = self.detectObjects(frame)
            # print(detections)
            if not detections:
                #In case object detection missed the object, give it another chance (5 chances, actually)
                if self.trackers is not None and self.timesMissed < 15:
                    self.timesMissed = self.timesMissed + 1
                    frame = self.updateTrackers(frame)
                    return frame

                self.timesMissed = 0
                self.trackers = []
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None
                return frame
            self.trackers = []
            for detection in detections:
                label = self.CLASSES[detection['idx']] + ' ' + str(detection['confidence'])
                box   = detection['box']
                color = self.objColor
                self.startTracker(frame, box, label, color)
                frame = self.updateFrame(frame, box, label, color)
            self.writeFrame(frame, fps)
            return frame
        frame = self.updateTrackers(frame)
        self.writeFrame(frame, fps)
        return frame

    def updateFrame(self, frame, box, label, color):
        """Draw a box around the detected object
        """
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        textSize, baseline = cv2.getTextSize( label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        # cv2.rectangle(frame, (box[0], box[1]) , (box[0] + textSize[0], box[1] - textSize[1] - 17), color, -1)
        # cv2.putText(frame, label, (box[0], box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame
    

    def writeFrame(self, frame, fps):
        nowDate = datetime.now().strftime("%Y-%m-%d")
        nowTime = datetime.now().strftime("%H-%M-%S")
        if self.writer is None:
            if not os.path.exists('F:/DoAn/data_video/' + nowDate):
                os.makedirs('F:/DoAn/data_video/' + nowDate)
            (h, w) = frame.shape[:2]
            self.writer = cv2.VideoWriter('F:/DoAn/data_video/' + nowDate + '/ai-' + nowTime + '.mp4',self.fourcc, fps, (w, h), True)
            try:
                retval, buffer = cv2.imencode('.jpg', frame)
                base64Image = base64.b64encode(buffer).decode('utf-8')
                
                image = '{\"name\": \"ai-' + nowTime + '.mp4\", \"data\": \"' + base64Image + '\"}'
                image = json.dumps(image)
                image = json.loads(image)
                headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                requests.post('http://localhost:8000/api/image-verify/async-verify', data=image, headers=headers)
                # requests.get('http://localhost:8000/api/video?name=ai-{}.mp4'.format(nowTime))
            except:
                print("Cannot connect to server")
        self.writer.write(frame)

    def detectObjects(self, frame):
        (h, w) = frame.shape[:2]
        fx = 300
        fy = 300
        mean   = cv2.mean(frame)

        #Make input image square to avoid geometric distortions
        frame  = cv2.copyMakeBorder(frame, 0, w - h, 0, 0, cv2.BORDER_CONSTANT, round(max(mean)))
        blob   = cv2.dnn.blobFromImage(frame,  1/max(mean), (fx, fy), mean, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        detected   = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.95 and int(detections[0, 0, i, 1]) == 15:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, w, w, w])
                detected.append({'box':box.astype("int"), 'confidence': confidence, 'idx': idx})
        return detected

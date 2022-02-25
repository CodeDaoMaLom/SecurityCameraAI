from threading import Thread
import cv2, time
import requests
import base64
import json
import os

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
anterior = 0

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/60
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):        
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('frame', self.frame)

        cv2.waitKey(self.FPS_MS)

if __name__ == '__main__':
    src = 'http://192.168.0.7:5001/stream.mjpg'
    threaded_camera = ThreadedCamera(src)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass
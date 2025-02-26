import cv2
import threading
from queue import Queue
import os

class camera():

    def __init__(self,cameraindex=0,preview=False):
        self.preview=preview
        self.WIDTH=1280
        self.HEIGHT=720
        self.cameraindex=cameraindex
        self.STREAM_RESOLUTION = str(self.WIDTH)+"x"+str(self.HEIGHT)

      
    def preview_capture(self,frame):
        cv2.imshow("video", frame)
        #frame = self.paint_frame(frame)

    def get_frames(self):
        cap = cv2.VideoCapture(self.cameraindex)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 24)  # we read the stream at 30 fps

        while cap.isOpened():
            ret, self.frame = cap.read()
            
            if not ret:
                break
            
            # Apply a blur effect
            strength=91 # use only odd numbers
            blurred_frame = cv2.GaussianBlur(self.frame, (strength, strength), 0)  # Change (15, 15) to adjust blur strength
            
            if self.preview:
                # Display the blurred frame
                cv2.imshow('frame', blurred_frame)
                cv2.waitKey(1)
            
            # Use the blurred frame for further processing
            self.put_with_drop(blurred_frame)
            # self.frame_queue.put(blurred_frame)

        cap.release()
    
    
    def start_cam(self):
        self.frame_queue = Queue(maxsize=300)  # <-- Set a size limit for the queue

        capture_thread = threading.Thread(target=self.get_frames)
        capture_thread.start()
        #capture_thread.join()
        #cam_thread = threading.Thread(target=self.get_frames, args=(self.frame_queue,))
        #cam_thread.start()
        #cam_thread.join()

        
    def put_with_drop(self, item):
        if self.frame_queue.full():
            self.frame_queue.get()  # this removes the first item
        self.frame_queue.put(item)
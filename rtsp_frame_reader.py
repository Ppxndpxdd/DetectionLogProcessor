import logging
import time
import cv2
import threading

class RTSPFrameReader(threading.Thread):
    """
    Opens a persistent connection to the RTSP stream and continuously updates the latest frame.
    """
    def __init__(self, rtsp_url: str):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            logging.error(f"RTSPFrameReader: Failed to open RTSP stream: {rtsp_url}")
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                logging.error("RTSPFrameReader: Failed to read frame, trying to reconnect...")
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                return None

    def stop(self):
        self.stopped = True
        self.cap.release()
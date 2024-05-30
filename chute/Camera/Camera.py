import cv2, queue, threading


class Camera:
    """
    Video Capture object that outputs the latest frame of the source

    Attributes:
        source (str): RTSP stream url, camera source
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if str(source).split(".")[-1] in ["mp4", "avi"]:
            self.video_type = "local"
        else:
            self.q = queue.Queue()
            t = threading.Thread(target=self._reader)
            t.daemon = True
            t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        """return latest frame from camera"""
        if self.video_type == "local":
            return self.cap.read()[1]
        else:
            return self.q.get()


if __name__ == "__main__":
    camera = Camera("rtsp://cs2projs:cs2projs@192.168.0.166/stream1")
    while True:
        frame = camera.read()
        cv2.imshow("window", frame)
        if chr(cv2.waitKey(1) & 255) == "q":
            break

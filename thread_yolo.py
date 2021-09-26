import threading
from collections import deque
import cv2
import time
import numpy as np

from flask import Flask, Response

app = Flask(__name__)
disp_frame = dict()

@app.route('/')
def server_stream():
    return Response(getFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def getFrames():
    global disp_frame
    while True:
        for name, frame in disp_frame:
            print(name)
            # if frame is not None:
            #     # disp_frame -> jpeg 인코딩 (품질 100)
            #     ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            #     bframe = jpeg.tobytes()
            #     yield (b'--frame\r\n'
            #         b'Content-Type: image/jpeg\r\n\r\n' + bframe + b'\r\n\r\n')
            # else:
            #     yield (b'--frame\r\n'
            #         b'Content-Type: image/jpeg\r\n\r\n\r\n\r\n')


# 영상 입력
class ThreadA(threading.Thread):
    def __init__(self, url=None, name=None):
        threading.Thread.__init__(self)
        self.name = name
        self.cap = None
        self.frame = None

        if url is not None:
            width = 640
            height = 480
            self.cap = cv2.VideoCapture(url)
            self.cap.set(3, width)
            self.cap.set(4, height)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
    



# 영상 detection
class ThreadB(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        pass


# 영상 출력
class ThreadC(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global disp_frame
        fps = dict()
        start_time = time.time()
    
        while True:
            for th_read in thread_list:
                if th_read.frame is not None:
                    frame = th_read.frame.copy()
                    # frame = cv2.resize(frame, dsize=(640,480), interpolation=cv2.INTER_AREA)
                    cv2.imshow(str(th_read.name), frame)

                    disp_frame[th_read.name] = frame
                    
                    if th_read.name in fps:
                        fps[th_read.name] += 1
                    else:
                        fps[th_read.name] = 0
                    
            if time.time()-start_time > 1:
                for name, count in fps.items():
                    print('FPS: ' + name, count)
                    fps[name] = 0
                    start_time = time.time()

            if cv2.waitKey(1) & 0xff==ord('q'):
                break
                
        cv2.destroyAllWindows()


            


if __name__=="__main__":
    cctv = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'

    thread_list = []
    thread_list.append(ThreadA(url=0, name='webcam'))
    # thread_list.append(ThreadA(url=1, name='cctv'))
    for thread in thread_list:
        thread.start()

    th_view = ThreadC()
    th_view.start()

    app.run(host='0.0.0.0', debug=False, port=5000)
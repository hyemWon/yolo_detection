import threading
import time
import cv2
import numpy as np
from collections import deque
import yolo

# from flask import Flask, Response
#
# app = Flask(__name__)
# disp_frame = None
#
# @app.route('/')
# def server_stream():
#     return Response(getFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# def getFrames():
#     global disp_frame
#
#     while True:
#         if disp_frame is not None:
#             # disp_frame -> jpeg 인코딩 (품질 100)
#             ret, jpeg = cv2.imencode('.jpg', disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#             bframe = jpeg.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + bframe + b'\r\n\r\n')
#         else:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n\r\n\r\n')





# 영상 입력
class ThreadA(threading.Thread):
    # disp_frame = []
    result = []

    def __init__(self, url=None, name=None):
        threading.Thread.__init__(self)
        self.cap = None
        self.name = name

        self.results = []
        self.frame = None
        self.frame_cnt = 0

        if url is not None:
            width = 640
            height = 480
            self.cap = cv2.VideoCapture(url)
            self.cap.set(3, width)
            self.cap.set(4, height)

        #ThreadC.start_t[self.name] = None
        #ThreadC.count[self.name] = 0

    def run(self):
        frame_cnt = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.frame_cnt += 1

                if (self.frame_cnt%3 == 0) and (len(th_detector.deque) < 50):
                    th_detector.deque.append([self, frame])

                #self.disp_frame = (frame, self.name)

            #time.sleep(1/fps)



class ThreadB(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.deque = deque()

    def run(self):
        while True:
            if len(self.deque) > 0:
                th_read, frame = self.deque.popleft()
                #~~~~
                results = []
                th_read.results = results

            time.sleep(0.0001)
            '''
            if len(self.deque) > 0:
                frame = self.deque.popleft()
                frame[0] = cv2.resize(frame[0], dsize=(640, 480), interpolation=cv2.INTER_AREA)
                ThreadA.result = yolo.net.detect(frame[0])
            '''




# 영상 출력
class ThreadC(threading.Thread):
    start_t = dict()
    count = dict()

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            for th_read in thread_list:
                if th_read.frame is not None:
                    frame = th_read.frame.copy()

                    #for track in th_read.results:
                    #   draw frame~

                    #name, image = th_read1.disp_frame[-1], th_read1.disp_frame[0]
                    frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                    cv2.imshow(th_read.name, frame)

                    '''
                    self.count[name] += 1
    
                    if self.start_t[name] is None:
                        self.start_t[name] = time.time()
    
                    if time.time() - self.start_t[name] > 1:
                        print(name, self.count[name])
                        self.count[name] = 0
                        self.start_t[name] = time.time()
                    '''

            if cv2.waitKey(1) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                exit()

            time.sleep(1/30)




if __name__ == '__main__':
    cctv = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'

    th_detector = ThreadB()
    th_detector.start()

    thread_list = []
    thread_list.append(ThreadA(url=0, name='webcam'))
    thread_list.append(ThreadA(url=cctv, name='cctv1'))
    for thread in thread_list:
        thread.start()

    #th_read1 = ThreadA(url=cctv, name='cctv')
    #th_read1.start()
    th_view = ThreadC()
    th_view.start()

    '''
    thread_list.append(ThreadB())
    thread_list.append(ThreadC())
    thread_list.append(ThreadA(url=cctv, name='cctv', thread=thread_list[0]))
    thread_list.append(ThreadA(url=0, name='webcam', thread=thread_list[0]))


    for thread in thread_list:
        thread.start()
    '''

    # app.run(host='0.0.0.0', debug=False, port=5000)
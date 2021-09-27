import threading
from collections import deque
import cv2
import time
import numpy as np
import yolo

from flask import Flask, Response, render_template

app = Flask(__name__)
disp_frame = dict()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cctv')
def cctv():
    return Response(getFrames('cctv'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return Response(getFrames('webcam'), mimetype='multipart/x-mixed-replace; boundary=frame')


def getFrames(name):
    global disp_frame
    while True:
        if disp_frame[name] is not None:
            # disp_frame -> jpeg 인코딩 (품질 100)
            ret, jpeg = cv2.imencode('.jpg', disp_frame[name], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            bframe = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bframe + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n\r\n\r\n')


# 영상 입력
class ThreadA(threading.Thread):
    def __init__(self, url=None, name=None):
        threading.Thread.__init__(self)
        self.name = name
        self.cap = None

        self.frame = None
        self.frame_cnt = 0
        self.results = []
        self.fps = 0

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
                frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                self.frame = frame
                self.frame_cnt += 1

                if (self.frame_cnt % 3 == 0) and (len(th_detector.deque) < 50):
                    th_detector.deque.append([self, frame])

            time.sleep(1/30)


# 영상 detection
class ThreadB(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.deque = deque()

    def run(self):
        while True:
            if len(self.deque) > 0:
                th_read, frame = self.deque.popleft()
                results = yolo.net.detect(frame)
                th_read.results = results

            time.sleep(0.0001)



# 영상 출력
class ThreadC(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global disp_frame
        global pts
        global mask
        global roi_start

        fps = 0
        count = 0
        start_time = time.time()

        while True:
            for th_read in thread_list:
                if th_read.frame is not None:
                    frame = th_read.frame.copy()

                    for detection in th_read.results:
                        label = detection[0]
                        confidence = detection[1]
                        x, y, w, h = detection[2]
                        xmin, ymin = int(x - w / 2), int(y - h / 2)
                        xmax, ymax = int(x + w / 2), int(y + h / 2)

                        pstring = label + ": " + str(np.rint(100 * confidence)) + "%"

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                        cv2.putText(frame, pstring, (xmin, ymin - 12), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

                    # ROI 영역
                    if th_read.name in pts:
                        pass
                    else:
                        pts[th_read.name] = []
                        roi_start[th_read.name] = False

                    if len(pts[th_read.name]) > 0:
                        cv2.circle(frame, pts[th_read.name][-1], 3, (0, 0, 0), -1)
                    if len(pts[th_read.name]) > 1:
                        for i in range(len(pts[th_read.name]) - 1):
                            cv2.circle(frame, pts[th_read.name][i], 3, (0, 0, 0), -1)
                            cv2.line(frame, pts[th_read.name][i], pts[th_read.name][i + 1], (0, 0, 0), 2)


                    cv2.putText(frame, str(fps), (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow(str(th_read.name), frame)
                    cv2.setMouseCallback(str(th_read.name), draw_roi, th_read)


                    if roi_start[th_read.name]:
                        disp_frame[th_read.name] = cv2.bitwise_and(frame, mask[th_read.name])
                    else:
                        disp_frame[th_read.name] = frame


                    count += 1

            if time.time() - start_time >= 1:
                fps = int(count / len(thread_list))
                print('FPS: ', fps)
                count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            time.sleep(1/30)

        cv2.destroyAllWindows()



def draw_roi(event, x, y, flags, param):
    global pts
    global mask
    global roi_start

    # 왼쪽 마우스 클릭
    if event == cv2.EVENT_LBUTTONDOWN:
        pts[param.name].append((x,y))
        print(pts)
    # 오른쪽 마우스 클릭
    if event == cv2.EVENT_RBUTTONDOWN:
        pts[param.name].pop()
    # 중간 마우스 클릭
    if event == cv2.EVENT_MBUTTONDOWN:
        if (not roi_start[param.name]) and len(pts[param.name]) > 2:
            roi_start[param.name] = True
            mask[param.name] = np.zeros(param.frame.shape, np.uint8)
            points = np.array(pts[param.name], np.int32)
            mask[param.name] = cv2.fillPoly(mask[param.name], [points], (255,255,255))
            print(mask[param.name])
        # ROI 해제
        else:
            roi_start[param.name] = False
            pts[param.name].clear()





if __name__ == "__main__":
    pts = dict()
    mask = dict()
    roi_start = dict()

    cctv = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'

    th_detector = ThreadB()
    th_detector.start()

    thread_list = []
    thread_list.append(ThreadA(url=0, name='webcam'))
    thread_list.append(ThreadA(url=cctv, name='cctv'))
    for thread in thread_list:
        thread.start()

    th_view = ThreadC()
    th_view.start()

    app.run(host='0.0.0.0', debug=False, port=5000)

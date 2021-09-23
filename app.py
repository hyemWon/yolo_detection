import threading
import time
import cv2
import numpy as np
from collections import deque
import yolo_1

from flask import Flask, Response

app = Flask(__name__)
disp_frame = None

@app.route('/')
def server_stream():
    return Response(getFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def getFrames():
    global disp_frame

    while True:
        if disp_frame is not None:
            # disp_frame -> jpeg 인코딩 (품질 100)
            ret, jpeg = cv2.imencode('.jpg', disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            bframe = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bframe + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n\r\n\r\n')





# 영상 입력
class ThreadA(threading.Thread):
    def __init__(self, url=None, name=None, thread=None):
        threading.Thread.__init__(self)
        self.cap = None
        self.name = name
        self.threadB = thread

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
                if len(self.threadB.input_deque) < 30:
                    self.threadB.input_deque.append([frame, self.name])



# 영상 사람 detection
class ThreadB(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.input_deque = deque()
        self.output_deque = deque()

    def run(self):
        while True:
            if len(self.input_deque) > 0:
                frame = self.input_deque.popleft()
                frame[0] = cv2.resize(frame[0], dsize=(640, 480), interpolation=cv2.INTER_AREA)
                result = yolo_1.net.detect(frame[0])

                for detection in result:
                    label = detection[0]
                    confidence = detection[1]
                    x, y, w, h = detection[2]
                    xmin, ymin = int(x - w / 2), int(y - h / 2)
                    xmax, ymax = int(x + w / 2), int(y + h / 2)

                    pstring = label + ": " + str(np.rint(100 * confidence)) + "%"

                    cv2.rectangle(frame[0], (xmin, ymin), (xmax, ymax), (0,255,0), 3)
                    cv2.putText(frame[0], pstring, (xmin, ymin-12), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 1)

                    self.output_deque.append(frame)



#  영상 출력
class ThreadC(threading.Thread):


    def __init__(self, thread=None):
        threading.Thread.__init__(self)
        self.threadB = thread

    def run(self):
        global disp_frame
        start_t = time.time()
        count = 0

        while True:
            if len(self.threadB.output_deque) > 0:
                frame = self.threadB.output_deque.popleft()
                cv2.imshow(str(frame[1]), frame[0])
                count += 1

                disp_frame = frame[0]

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    exit()

            if time.time() - start_t > 1:
                print(frame[1], count)
                count = 0
                start_t = time.time()








if __name__ == '__main__':
    cctv = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
    thread_list = []

    thread_list.append(ThreadB())
    thread_list.append(ThreadC(thread=thread_list[0]))
    thread_list.append(ThreadA(url=cctv, name='cctv', thread=thread_list[0]))

    for thread in thread_list:
        thread.start()

    app.run(host='0.0.0.0', debug=False, port=5000)
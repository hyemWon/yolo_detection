import socket
import threading
import cv2
import numpy as np
import time
from collections import deque
import yolo

from flask import Flask, Response, render_template

disp_frame = dict()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    # return Response(getFrames('1'), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/cctv')
# def cctv():
#     return Response(getFrames('1'), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/webcam')
# def webcam():
#     return Response(getFrames('0'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/<int:video_id>')
def videoParser(video_id):
    print(type(video_id), video_id)
    return Response(getFrames(video_id), mimetype='multipart/x-mixed-replace; boundary=frame')


def getFrames(name):
    global disp_frame
    print(type(name))
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




class TCPServer(threading.Thread):
    def __init__(self, ip, port):
        threading.Thread.__init__(self)
        self.thread_list = []

        self.address = (ip, port)
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.sock.bind(self.address)
        self.sock.listen()

    def run(self):
        print("[STARTING] Server is starting...")
        while True:
            conn, addr = self.sock.accept()
            print(f"[NEW CONNECTION] {addr} connected.")
            self.thread_list.append(ThreadReceive(conn, addr))
            self.thread_list[-1].start()

        self.sock.close()


# 영상 수신
class ThreadReceive(threading.Thread):
    def __init__(self, conn=None, addr=None):
        threading.Thread.__init__(self)
        self.conn = conn
        self.addr = addr
        self.name = None
        self.connected = True

        self.frame = None
        self.frame_cnt = 0
        self.results = []

        print(self.addr, ' 영상 수신 대기')

    def run(self):
        self.name = len(server.thread_list)
        print('이름: ', self.name)
        print('타입: ', type(self.name))

        while self.connected:
            try:
                length = self.recvall(16)
                string_data = self.recvall(int(length))
                data = np.fromstring(string_data, dtype='uint8')

                self.frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                self.frame_cnt += 1
                
                if (self.frame_cnt % 3 == 0) and (len(th_detector.deque) < 50):
                    th_detector.deque.append([self, self.frame])

            except:
                self.connected = False

        cv2.destroyAllWindows()
        server.thread_list.remove(self)
        print('삭제')

        # 객체 이름 업데이트
        for idx in range(len(server.thread_list)):
            server.thread_list[idx].name -= 1
        print(server.thread_list)


    # 수신 버퍼를 읽어서 반환
    def recvall(self, count):
        buf = b''
        while count:
            newbuf = self.conn.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    

# 영상 검출
class ThreadDetect(threading.Thread):
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
        

# 영상 송출
class ThreadSend(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global pts
        global mask
        global roi_start
        global disp_frame

        fps = 0
        count = 0
        start_time = time.time()

        while True:
            for th_read in server.thread_list:
                if th_read.frame is not None:
                    frame = th_read.frame.copy()
                    
                    # 사람 박스 그리기
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
                    cv2.imshow(str(th_read.addr), frame)
                    cv2.setMouseCallback(str(th_read.addr), draw_roi, th_read)


                    if roi_start[th_read.name]:
                        disp_frame[th_read.name] = cv2.bitwise_and(frame, mask[th_read.name])
                    else:
                        disp_frame[th_read.name] = frame

                    # cv2.imshow('roi_' + str(th_read.name), disp_frame[th_read.name])


                    count += 1

                    if cv2.waitKey(1) & 0xff == ord('q'):
                        th_read.connected = False

            if time.time() - start_time >= 1:
                if server.thread_list:
                    fps = int(count/len(server.thread_list))
                    # fps = count
                    print('FPS: ', fps)
                    count = 0
                    start_time = time.time()

            time.sleep(1/30)

        cv2.destroyAllWindows()



# 관심 영역 마우스 콜백 함수
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
            # print(mask[param.name])
        # ROI 해제
        else:
            roi_start[param.name] = False
            pts[param.name].clear()





if __name__ == "__main__":
    pts = dict()
    mask = dict()
    roi_start = dict()


    SERVER = "192.168.0.69"
    PORT = 5050
    ADDR = (SERVER, PORT)
    HEADER = 64
    FORMAT = 'utf-8'

    th_detector = ThreadDetect()
    th_detector.start()

    server = TCPServer(SERVER, PORT)
    server.start()

    th_view = ThreadSend()
    th_view.start()

    app.run(host='0.0.0.0', debug=False, port=5000)
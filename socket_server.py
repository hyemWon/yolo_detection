import socket
import threading
import cv2
import numpy as np
import time
from collections import deque
import yolo




class TCPServer(threading.Thread):
    def __init__(self, ip, port):
        threading.Thread.__init__(self)
        self.thread_list = []

        self.address = (ip, port)
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #
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
        self.connected = True

        self.frame = None
        self.frame_cnt = 0
        self.results = []
        self.roi = None

        print(self.addr, ' 영상 수신 대기')

    def run(self):
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

        fps = 0
        count = 0
        start_time = time.time()

        while True:
            for th_read in server.thread_list:
                if th_read.frame is not None:
                    frame = th_read.frame.copy()
                    name = th_read.addr

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
                    if name in pts:
                        pass
                    else:
                        pts[name] = []
                        roi_start[name] = False

                    if len(pts[name]) > 0:
                        cv2.circle(frame, pts[name][-1], 3, (0, 0, 0), -1)
                    if len(pts[name]) > 1:
                        for i in range(len(pts[name]) - 1):
                            cv2.circle(frame, pts[name][i], 3, (0, 0, 0), -1)
                            cv2.line(frame, pts[name][i], pts[name][i + 1], (0, 0, 0), 2)


                    cv2.putText(frame, str(fps), (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow(str(name), frame)
                    cv2.setMouseCallback(str(name), draw_roi, th_read)


                    if roi_start[name]:
                        th_read.roi = cv2.bitwise_and(frame, mask[name])
                    else:
                        th_read.roi = frame

                    cv2.imshow('roi_' + str(name), th_read.roi)

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
        pts[param.addr].append((x,y))
    # 오른쪽 마우스 클릭
    if event == cv2.EVENT_RBUTTONDOWN:
        pts[param.addr].pop()
    # 중간 마우스 클릭
    if event == cv2.EVENT_MBUTTONDOWN:
        if (not roi_start[param.addr]) and len(pts[param.addr]) > 2:
            roi_start[param.addr] = True
            mask[param.addr] = np.zeros(param.frame.shape, np.uint8)
            points = np.array(pts[param.addr], np.int32)
            mask[param.addr] = cv2.fillPoly(mask[param.addr], [points], (255,255,255))
        # ROI 해제
        else:
            roi_start[param.addr] = False
            pts[param.addr].clear()








if __name__ == "__main__":
    disp_frame = dict()
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




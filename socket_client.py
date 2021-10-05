import socket
import threading
import cv2
import numpy as np
import time
import sys

class ClientSocket(threading.Thread):
    def __init__(self, server, port, url, header=64, format='utf-8'):
        threading.Thread.__init__(self)
        self.server = server
        self.port = port
        self.header = header
        self.format = format
        self.sock = None

        self.url = url
        self.fps = 0
        self.cap = None

        if url is not None:
            self.cap = cv2.VideoCapture(self.url)
            self.cap.set(3, 640)
            self.cap.set(4, 480)


    def run(self):
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.sock.connect((self.server, self.port))
        # 영상 송신
        self.send()

    def send(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # 이미지의 품질 설정

        start_time = time.time()
        count = 0
        while self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)

                    # 프레임을 string 형태로 인코딩
                    res, encode_frame = cv2.imencode('.jpg', frame, encode_param)

                    string_data = np.array(encode_frame).tostring()

                    # string 데이터 socket을 통해 전송
                    # self.sock.sendall(str(len(string_data)).encode().ljust(16))
                    # self.sock.sendall(string_data)
                    self.sock.sendall(str(len(string_data)).encode().ljust(16) + string_data)

                    cv2.putText(frame, str(self.fps), (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow(str(self.url), frame)

                    count += 1

                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break

                    if time.time() - start_time >= 1:
                        self.fps = int(count)
                        count = 0
                        start_time = time.time()

                    time.sleep(0.001)
            except:
                break
        cv2.destroyAllWindows()




def main():
    # URL = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'

    if len(sys.argv)==2:
        URL = sys.argv[1]
        if URL.isdigit():
            URL = int(URL)
    else:
        URL = 0

    SERVER = "192.168.0.69"
    PORT = 5050
    HEADER = 64
    FORMAT = 'utf-8'

    print(URL)

    client = ClientSocket(SERVER, PORT, URL, HEADER, FORMAT)
    client.start()

    # cap = cv2.VideoCapture(URL)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imshow('test', frame)
    #
    #     if cv2.waitKey(1) & 0xff == ord('q'):
    #         break
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
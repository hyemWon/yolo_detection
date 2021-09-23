import threading
import time
import cv2
from collections import deque

# 영상 입력
class ThreadA(threading.Thread):
    def __init__(self, url=None, name=None):
        threading.Thread.__init__(self)
        self.queue = deque()
        self.url = url
        self.stop = 0

        try:
            width = 640
            height = 480
            self.cap = cv2.VideoCapture(url)
            self.cap.set(3, width)
            self.cap.set(4, height)
        except:
            print('장치가 연결되지 않았습니다.')

    def run(self):
        while (True):
            ret, frame = self.cap.read()

            if ret:
                if len(self.queue) < 30:
                    self.queue.append(frame)

            # thread 중단
            if self.stop==1:
                self.cap.release()
                cv2.destroyAllWindows()
                break

# 영상 출력
class ThreadB(threading.Thread):
    def __init__(self, thread):
        threading.Thread.__init__(self)
        self.threadA = thread

    def run(self):
        start_t = time.time()
        count = 0
        while True:
            if len(self.threadA.queue) > 0:
                frame = self.threadA.queue.popleft()
                frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                cv2.imshow(str(self.threadA.url), frame)
                count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.threadA.stop = 1
                    break
            # fps 출력
            if time.time() - start_t > 1:
                print(self.threadA.url, count)
                count = 0
                start_t = time.time()


# 영상 사람 detection
class ThreadC(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.input_deque = deque()
        self.output_deque = deque()

    def run(self):
        pass





if __name__ == '__main__':
    cctv = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
    thread_list = []

    thread_list.append(ThreadA(url=0, name='webcam'))
    thread_list.append(ThreadA(url=cctv, name='cctv'))
    thread_list.append(ThreadB(thread=thread_list[0]))
    thread_list.append(ThreadB(thread=thread_list[1]))

    for thread in thread_list:
        thread.start()
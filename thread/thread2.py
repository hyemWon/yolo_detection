import threading
import time
import cv2
from collections import deque



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
        pass



#  영상 출
class ThreadC(threading.Thread):
    def __init__(self, thread=None):
        threading.Thread.__init__(self)
        self.threadB = thread

    def run(self):
        start_t = time.time()
        count = dict()
        while True:
            if len(self.threadB.input_deque) > 0:
                frame = self.threadB.input_deque.popleft()
                frame[0] = cv2.resize(frame[0], dsize=(640, 480), interpolation=cv2.INTER_AREA)
                cv2.imshow(str(frame[1]), frame[0])
                # count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    exit()

                if frame[1] not in count:
                    count[frame[1]] = 0
                else:
                    count[frame[1]] += 1

            if time.time() - start_t > 1:
                print(count)
                count['webcam'] = 0
                count['cctv'] = 0
                start_t = time.time()



if __name__ == '__main__':
    cctv = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
    thread_list = []

    thread_list.append(ThreadB())
    thread_list.append(ThreadC(thread=thread_list[0]))
    thread_list.append(ThreadA(url=1, name='webcam', thread=thread_list[0]))
    thread_list.append(ThreadA(url=cctv, name='cctv', thread=thread_list[0]))

    for thread in thread_list:
        thread.start()
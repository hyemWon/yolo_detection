# 파이썬에서 C의 데이터 타입이나, DLL 혹은 공유 라이브러리 사용 가능하도록
from ctypes import *
import numpy as np
import os



class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("best_class_idx", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)

hasGPU = True
lib = CDLL("./lib/libdarknet.so", RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]


predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class Yolo:
    net = None
    meta = None
    def __init__(self, config_file, weights, data_file, gpus=0, batch_size=1):
        set_gpu(gpus)

        self.net = load_net_custom(config_file.encode("utf-8"), weights.encode("utf-8"), 0, batch_size)
        self.meta = load_meta(data_file.encode("utf-8"))
        self.class_names = []

        # class 종류 읽기
        with open(data_file) as metaF:
            result = None
            for line in metaF:
                if 'names' in line:
                    result = line.split('=')[1].strip()
            if os.path.exists(result):
                with open(result, 'r') as namesF:
                    self.class_names = [line.strip() for line in namesF]


    def detect(self, frame, thresh=0.5, hier_thresh=0.5, nms=0.45):
        im, arr = self.array_to_image(frame)
        im_width = im.w
        im_height = im.h

        pnum = pointer(c_int(0))
        predict_image(self.net, im)

        detections = get_network_boxes(self.net, im_width, im_height, thresh, hier_thresh, None, 0, pnum, 0)

        num = pnum[0] # detect된 객체 수?
        do_nms_sort(detections, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if detections[j].prob[i] > 0:
                    b = detections[j].bbox
                    res.append((self.class_names[i], detections[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1]) # probability를 기준으로 내림차순 정렬

        # 메모리 해제
        del im, arr
        free_detections(detections, num)

        return res

    def array_to_image(self, arr):
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w,h,c,data)
        return im, arr


# def video_capture(net, url):
#     cap = cv2.VideoCapture(url)
#
#     fps = 0
#     fps_cnt = 0
#     fps_time = time.time()
#
#     if not cap.isOpened():
#         exit()
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             result = net.detect(frame)
#
#             fps_cnt += 1
#
#             for detection in result:
#                 label = detection[0]
#                 confidence = detection[1]
#                 x, y, w, h = detection[2]
#                 xmin, ymin = int(x-w/2), int(y-h/2)
#                 xmax, ymax = int(x+w/2), int(y+h/2)
#
#                 cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
#                 cv2.putText(frame, label, (xmin, ymin-12), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 1)
#
#             if time.time() - fps_time >= 1:
#                 fps = fps_cnt
#                 fps_cnt = 0
#                 fps_time = time.time()
#
#             cv2.putText(frame, 'FPS: {}'.format(fps), (15,25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,0), 2)
#             cv2.imshow('Person Detection', frame)
#
#             if cv2.waitKey(1) & 0xff == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()


net = Yolo('cfg/yolov4-custom.cfg', 'weights/yolov4-custom_best.weights', 'custom/obj.data', 0, 1)







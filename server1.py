import cv2

thres = 0.45
nms_threshold = 0.2

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture("rtsp://sathya08:sathya08@192.168.1.10:554/stream1")

    def __del__(self):
        self.video.releast()

    def get_frame(self):

        while True:
            ret, frame = self.video.read()
            classIds, confs, bbox = net.detect(frame, confThreshold=thres)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
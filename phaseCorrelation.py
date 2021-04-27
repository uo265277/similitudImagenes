import cv2
import math
import time
import numpy as np

class CorrelationCalculator(object):
    'TODO: class description'

    version = '0.1'

    def __init__(self, initial_frame, detection_threshold=4):
        self.initial_frame = np.float32(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY))
        self.detection_threshold = detection_threshold

    def detect_phase_shift(self, current_frame):
        'returns detected sub-pixel phase shift between two arrays'
        self.current_frame = np.float32(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
        shift = cv2.phaseCorrelate(self.initial_frame, self.current_frame)
        return shift

# implementation

import cv2

img = cv2.imread(r"/home/claudia/PycharmProjects/similitudImagenes/directorios/FRANK_PRUEBAS/B.jpg")
img2 = cv2.imread(r"/home/claudia/PycharmProjects/similitudImagenes/directorios/FRANK_PRUEBAS/B_MOD1.jpg")

obj = CorrelationCalculator(img)
print(obj)
shift = obj.detect_phase_shift(img2)

print(str(shift))
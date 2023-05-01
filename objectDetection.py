import os
import random
from IPython.display import Image  # for displaying images
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

detections_dir = "yolov5/runs/detect/yolo_basket_det/"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]

random_detection_image = Image.open(random.choice(detection_images))
plt.imshow(np.array(random_detection_image))
plt.show()
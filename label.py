import os
import cv2
import glob
import torch

def choose_labels(filename):
    # To show the detection use the line below
    detections.show()
    x = input('0/1/2: ')

    if x == '1':
        size = len(filename)
        name = filename[:size - 3]
        target = 'label/' + name + 'txt'
        with open(target, 'w') as f:

            for result in detections.xywh:
                for i in range(result.shape[0]):
                    # Ottieni la classe associata
                    class_index = int(result[i,5])
                    # Ottieni le coordinate del bounding box
                    x,y,w,h = result[i,0:4]/416
                    to_write = str(class_index) + ' ' + str(x.item()) + ' ' + str(y.item()) + ' ' + str(w.item()) + ' ' + str(h.item()) + '\n'
                    f.write(to_write)

    if x == '2':
        dest = 'bin/' + filename
        os.rename(image, dest)

    if x == '0':
        return True

images = glob.glob('dataset/train/images/*.jpg')
Path = 'C:/Users/Computer/Documents/GitHub/CV_BasketAnalysis/'
model = torch.hub.load(f'{Path}yolo5', 'custom',
                            path=f"{Path}yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt",source='local', force_reload=True)

for image in images:
    img = cv2.imread(image)
    # Inverti l'ordine dei canali da BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = model(img)
    filename = image.split('\\')[1]

    if detections.xywh.shape[0] == 1:
         for result in detections.xywh:
                # Ottieni le coordinate del bounding box
                for i in range(result.shape[0]):
                    # Ottieni la classe associata
                    class_index = int(result[i,5])
                    if class_index == '0':
                        dest = 'bin/' + filename
                        os.rename(image, dest)

    else:
        check = choose_labels(filename)
        if check:
            break
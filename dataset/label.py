import os
import cv2
import glob
import torch
import shutil

def sub_label(filename):
    # get the name of the new file where it will store the new labels
    size = len(filename)
    name = filename[:size - 3]
    target = 'dataset/train/checked_label/' + name + 'txt'

    with open(target, 'w') as f:
        for result in detections.xywh:
            for i in range(result.shape[0]):
                # Ottieni la classe associata
                class_index = int(result[i,5])
                # Ottieni le coordinate del bounding box
                x,y,w,h = result[i,0:4]/416
                to_write = str(class_index) + ' ' + str(x.item()) + ' ' + str(y.item()) + ' ' + str(w.item()) + ' ' + str(h.item()) + '\n'
                # write the results in the file
                f.write(to_write)
    dest = 'dataset/train/checked_images/' + filename
    shutil.move(image, dest)
    
def del_image(filename, image):
    dest = 'tdataset/rain/bin/' + filename
    os.rename(image, dest)

def keep_label(filename, image):
    destdir = 'dataset/train/checked_images/' + filename
    os.rename(image, destdir)

def switch(x, image):
    filename = image.split('\\')[1]
    if x == '0':
        return True
    if x == '1':
        sub_label(filename)
    if x == '2':
        del_image(filename, image)
    if x == '3':
        keep_label(filename, image)
    return False

images = glob.glob('dataset/train/images/*.jpg')
model = torch.hub.load('yolo5', 'custom',
                            path="yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt",source='local', force_reload=True)

for image in images:
    check = False
    img = cv2.imread(image)
    # Inverti l'ordine dei canali da BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = model(img)
    filename = image.split('\\')[1]

    for result in detections.xywh:
        # Ottieni le coordinate del bounding box
        print(result.shape[0])
        if result.shape[0] <= 1:
            for i in range(result.shape[0]):
                # Ottieni la classe associata
                class_index = int(result[i,5])
                if class_index == 0:
                    check = switch(x='2',image=image)
                else:
                    check = switch(x='3', image=image)

        else:
            detections.show()
            # insert 0 to stop
            # insert 1 to substitute with the new label
            # insert 2 to move the image in the bin folder
            # insert 3 to keep the old label
            x = input('0/1/2/3: ')
            check = switch(x, image)
    if check:
        break

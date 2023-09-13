import cv2
from ultralytics import YOLO
import numpy as np

Path = 'distance/'
model_path = '../yolov8s_final/weights/best.pt'
sens1 = 0.065 # distance from camera update sensibility
sens2 = 0.045 # distance between r and p update sensibility

"""
This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using
MEASURED_DISTANCE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
:param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
:param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
:param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
:retrun Focal_Length(Float):
"""
def FocalLength(measured_distance, real_width, ref_path):
    # Carica il modello
    model = YOLO(model_path)
    # reading reference image from directory
    ref_image = cv2.imread(Path+ref_path)
    
    results = model(ref_image, conf=0.4)

    width_in_ref_image = 0
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        count_bb = r.boxes.cls.tolist().count(0.0)
        if count_bb == 1:
            #print(r.boxes.data.tolist())
            # Itera su ogni riga del tensore
            for row in r.boxes:
                #print(row.data.tolist()[0][-1])
                #print(row.data.tolist()[0])
                if row.data.tolist()[0][-1] == 0.0:
                    #print(row.xywh.tolist()[0])
                    # Crea una lista per la riga corrente e aggiungi i valori
                    width_in_ref_image = row.xywh.tolist()[0][2]
    print(width_in_ref_image)
    focal_length = (width_in_ref_image * measured_distance) / real_width
    print(focal_length)
    return focal_length

def check_intersection(a, b):
    x1 = max(min(a[0], a[2]), min(b[0], b[2]))
    y1 = max(min(a[1], a[3]), min(b[1], b[3]))
    x2 = min(max(a[0], a[2]), max(b[0], b[2]))
    y2 = min(max(a[1], a[3]), max(b[1], b[3]))
    if x1<x2 and y1<y2:
        return 1
    else:
        return 0

def getHeight(results, focal_length, real_width):
    for r in results:
        #print(r.boxes.cls.tolist())
        count_bb = r.boxes.cls.tolist().count(0.0)
        count_people = r.boxes.cls.tolist().count(1.0)
        if count_bb == 1 and count_people == 1:
            boxes=[]
            #print(r.boxes.data.tolist())
            # Itera su ogni riga del tensore
            for row in r.boxes:
                #print(row.data.tolist()[0])
                #print(row.data.tolist()[0][-1])
                #print(row.xywh.tolist()[0][:])
                if row.data.tolist()[0][-1] != 2.0:
                    # Crea una lista per la riga corrente e aggiungi i valori
                    riga_box = row.data.tolist()[0][:]
                    # Aggiungi la lista alla lista 'boxes'
                    boxes.append(riga_box)
                
            #print(f'{check_intersection(boxes[0],boxes[1])} check_intersection')
            #print(boxes)

            if check_intersection(boxes[0],boxes[1]):
                h_pp_img = [box[3]-box[1] for box in boxes if box[-1] == 1.0]
                w_bb_img = [box[2]-box[0] for box in boxes if box[-1] == 0.0]
                bb_dist = focal_length * real_width[0] / w_bb_img[0]
                #print(f'{h_pp_img} h_pp_img')
                #print(f'{w_bb_img} w_bb_img')
                #print(f'{bb_dist} bb_dist')
                return (h_pp_img[0] * bb_dist) / focal_length
        return 0

def updateHeight(results, focal_length, real_width):
    h = getHeight(results, focal_length, real_width)
    #print(f'{h} altezza')
    if h < 1.5 or h >= 2.4:
        return real_width
    else:
        real_width[3] += 1
        real_width[1] = (real_width[1] * (real_width[3] - 1)  + h) / real_width[3]
        return real_width

def updateCamDist(real_distance, d, cls):
    #print(f'{d,cls} d, cls')
    if abs(d - real_distance[cls]) > (sens1 * (real_distance[3+cls] + 1)) and real_distance[cls] != 0:
        #print(f'{np.sign(real_distance[cls] - d)} np.sign(real_distance[cls] - d)')
        real_distance[cls] += np.sign(d - real_distance[cls]) * (sens1 * (real_distance[3+cls] + 1))
    else:
        real_distance[cls] = d
    return real_distance

def updateDistance(distance, d):
    print(f'{distance,d} distance, d')
    if abs(d - distance[0]) > (sens2 * (distance[1] + 1)) and distance[0] != 0:
        #print(f'{np.sign(real_distance[cls] - d)} np.sign(real_distance[cls] - d)')
        distance[0] += np.sign(d - distance[0]) * (sens2 * (distance[1] + 1))
    else:
        distance[0] = d
    return distance
   

#fl = Utils.FocalLength(100, 25, 'ref.jpg')
#print(fl)
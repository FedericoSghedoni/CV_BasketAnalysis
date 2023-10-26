import cv2
from ultralytics import YOLO
import numpy as np

Path = 'distance/'
model_path = '../yolov8s_final/weights/best.pt'
sens1 = 0.065 # distance from camera update sensibility
sens2 = 0.045 # distance between r and p update sensibility
sens3 = 0.055

"""
This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using
MEASURED_DISTANCE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
:param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
:param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
:param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
:retrun Focal_Length(Float):
"""
def FocalLength(measured_distance, bb_width, ref_path):
    # Carica il modello
    model = YOLO(model_path)
    # reading reference image from directory
    ref_image = cv2.imread(Path+ref_path)
    
    results = model(ref_image, conf=0.4, verbose=False)

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
    # print(width_in_ref_image)
    focal_length = (width_in_ref_image * measured_distance) / bb_width
    # print(focal_length)
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

def updateDistance(old_d, new_d, count):
    #print(f'{distance,d} distance, d')
    if abs(new_d - old_d) > (sens2 * count) and old_d != 0:
        #print(f'{np.sign(real_distance[cls] - d)} np.sign(real_distance[cls] - d)')
        d = old_d + np.sign(new_d - old_d) * (sens3 * count)
    else:
        d = new_d
    return d

def checkRoi(pp_data, emb):
    for key in pp_data.keys():
        # Calcola la distanza euclidea
        eucl_dist = np.linalg.norm(emb - pp_data[key][0])
        print(f'{eucl_dist} eucl_dist')
        if eucl_dist < 8:
            return key
    return 0
    
def Embedding(roi):
    emb = np.array([])
    # Crea un modello di colore basato sulla media dei colori nella ROI
    emb = np.append(emb, np.mean(roi, axis=(0, 1)))
    return emb

def updateData(pp_data, roi, h):
    id = '001'
    emb = Embedding(roi)
    if len(pp_data) == 0:
        pp_data[id] = []
        pp_data[id].append(emb)
        for _ in range(3):
            pp_data[id].append([0, 0])
    else:
        idm = checkRoi(pp_data, emb)
        if idm:
            id = idm
            pp_data[id][0] = emb
        else:  
            max_key = max(pp_data.keys())
            id = str(int(max_key) + 1).zfill(3)
            pp_data[id] = []
            pp_data[id].append(emb)
            for _ in range(3):
                pp_data[id].append([0, 0])
    print(f'{id,h} id, h')
    if h >= 1.5 and h < 2.3:
        pp_data[id][1][1] += 1
        pp_data[id][1][0] = (pp_data[id][1][0] * (pp_data[id][1][1] - 1)  + h) / pp_data[id][1][1]      
    return pp_data, id

#fl = Utils.FocalLength(100, 25, 'ref.jpg')
#print(fl)
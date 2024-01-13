import csv
import math
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ultralytics import YOLO
import numpy as np
import seaborn as sns

Path = 'distance/'
model_path = '../yolov8s_final/weights/best.pt'
sens1 = 0.065 # distance from camera update sensibility
sens2 = 0.045 # distance between r and p update sensibility
sens3 = 0.055

pesi = np.array([1, 1, 1, 1, 1, 1, 0.50, 0.50])
sens_w = 0.04

max_dist = {'001': 0, '002' : 0, '003' : 0, '004' : 0, '005' : 0, '006' : 0}

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
        count_bb = r.boxes.cls.tolist().count(0.0)
        if count_bb == 1:
            # Itera su ogni riga del tensore
            for row in r.boxes:
                if row.data.tolist()[0][-1] == 0.0:
                    # Crea una lista per la riga corrente e aggiungi i valori
                    width_in_ref_image = row.xywh.tolist()[0][2]
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
    if abs(new_d - old_d) > (sens2 * count) and old_d != 0:
        d = old_d + np.sign(new_d - old_d) * (sens3 * count)
    else:
        d = new_d
    return d

def checkRoi(pp_data, emb):
    # Inizializza la distanza minima con un valore grande
    dist_min, mdist_min, sdist_min = float('inf'), float('inf'), float('inf')
    key_min, mkey_min, skey_min = None, None, None
    for key, val in {k : v for k, v in pp_data.items() if v[2][1] != 0}.items():
        
        if val[2][1] > 100 and not(emb[6] < 150 or emb[6] > 570): #controllo su contatore e su posizione della roi
            continue
            
        # Calcola la distanza euclidea
        diff = abs(emb - val[0][0])
        diff_m = abs(emb[:-2] - val[0][1])
        diff_w = abs(diff * (pesi ** (val[2][1] * sens_w)))
        
        eucl_dist = np.linalg.norm(diff_w)
        mean_dist = np.linalg.norm(diff_m)
        sum_dist = eucl_dist + mean_dist / 3.2
        #print(f'{val[2][1]} count {key}')
        print(f'{eucl_dist} eucl_dist {mean_dist} mean_eucl {sum_dist} sum_dist {key}')
        if eucl_dist < dist_min:
            dist_min = eucl_dist
            key_min = key
        if mean_dist < mdist_min:
            mdist_min = mean_dist
            mkey_min = key
        if sum_dist < sdist_min:
            sdist_min = sum_dist
            skey_min = key
    if key_min != mkey_min:
        print(f'{key_min, mkey_min, skey_min} key_min, mkey_min, skey_min')
    if sdist_min < 80: # Soglia della distanza
        if max_dist[key_min] < sdist_min:
            max_dist[key_min] = sdist_min
            #print(f'{max_dist} max dist')  

        return skey_min
    else:
        print(f'{skey_min, sdist_min} skey_min, sdist_min')
        return 0
    
def Embedding(roi): 
    emb = np.array([])
    # Crea un modello di colore basato sulla media dei colori nella ROI
    emb = np.append(emb, np.mean(roi[0], axis=(0, 1)))

    # Converte la ROI in HSV
    hsv_roi = cv2.cvtColor(roi[0], cv2.COLOR_BGR2HSV)

    # Aggiunge il tono dominante
    emb = np.append(emb, np.mean(hsv_roi[:, :, 0]))

    # Aggiunge la saturazione media 
    emb = np.append(emb, np.mean(hsv_roi[:, :, 1]))

    # Aggiunge il valore medio
    emb = np.append(emb, np.mean(hsv_roi[:, :, 2]))

    # Aggiunge le coordinate del centro della ROI
    emb = np.append(emb, roi[1])
    
    return emb

def updateData(pp_data, roi, h):
    id = '001'
    emb = Embedding(roi)
    if len(pp_data) == 0:
        pp_data[id] = []
        pp_data[id].append([emb, emb[:-2]])
        for _ in range(3):
            pp_data[id].append([0, 0])
    else:
        idm = checkRoi(pp_data, emb)
        if idm:
            id = idm
            pp_data[id][0][0] = emb
            pp_data[id][0][1] = (pp_data[id][0][1] * 0.97 + emb[:-2]) / 2 #
        else:  
            max_key = max(pp_data.keys())
            id = str(int(max_key) + 1).zfill(3)
            pp_data[id] = []
            pp_data[id].append([emb, emb[:-2]])
            for _ in range(3):
                pp_data[id].append([0, 0])
    if h >= 1.5 and h < 2.3:
        pp_data[id][1][1] += 1
        pp_data[id][1][0] = (pp_data[id][1][0] * (pp_data[id][1][1] - 1)  + h) / pp_data[id][1][1]      
    return pp_data, id

def calculate_angle(point1, point2, point3):
    # Calcola il vettore tra point2 e point1
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    norm1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    norm2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    if norm1 != 0 and norm2 != 0:
        # Calculate the value for acos
        value = dot_product / (norm1 * norm2)

        # Check if the value is within the valid range for arc cosine
        if -1 <= value <= 1:
            angle_radians = math.acos(value)
        else:
            angle_radians = 0
        angle_degrees = math.degrees(angle_radians)
        
    else: angle_degrees = 0

    return angle_degrees / 180 # Normaalized Value

class Buffer:
    def __init__(self, max_length):
        self.stack = []
        self.max_length = max_length

    def push(self, item):
        self.stack.append(item)
        if len(self.stack) > self.max_length:
            self.stack.pop(0)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return None

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)
    
    def clear(self):
        self.stack = []

def report(csv_file,data):
    # Write data to the CSV file
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        existing_data = list(csvreader)
    existing_data.append(data)
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(existing_data)

def result_graph(result_path, true_labels_train, true_labels_test, predictions_train, predictions_test):
    df = pd.read_csv(f'{result_path}train_output.csv')
    plt.figure()
    sns.lineplot(data=df,x='epoch',y='loss')

    df_test = pd.read_csv(f'{result_path}test_output.csv')
    sns.lineplot(data=df_test,x='epoch',y='loss')
    plt.savefig(f'{result_path}test_result.png')

    # Calculate confusion matrix
    plt.figure()
    cm_train = confusion_matrix(true_labels_train, predictions_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['canestro','fuori'])
    disp.plot()
    plt.savefig(f'{result_path}cm_train.png')

    # Calculate confusion matrix
    plt.figure()
    cm_test = confusion_matrix(true_labels_test, predictions_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['canestro','fuori'])
    disp.plot()
    plt.savefig(f'{result_path}cm_test.png')
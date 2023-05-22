import cv2
import numpy as np
import torch
from yolo5.models.experimental import attempt_load
from yolo5.utils.general import non_max_suppression
from deep_sort.deep_sort_app import run

# Inizializza il tracker DeepSORT
ThresholdDirection = 'somethings'
ThresholdVelocity = 'somethings'
Path = 'C:/Users/Computer/Documents/GitHub/CV_BasketAnalysis/'

def detect_image(model, img, conf_thres=0.3, iou_thres=0.3):
    model.eval()

    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.to(device)

    with torch.no_grad():
        img = img.reshape(1, 3, 640, 352)
        print(img.shape)
        pred = model(img)
        print(pred.shap)

    pred = non_max_suppression(pred, conf_thres, iou_thres)
    detections = []
    for det in pred[0]:
        if det is not None:
            det[:, :4] = det[:, :4].clone().cpu()
            detections.append(det)

    return detections

# Funzione per rilevare e tracciare gli oggetti nel frame
def track_objects(frame):
    # Esegue la rilevazione degli oggetti utilizzando il modello addestrato YOLOv5
    # e ottiene le bounding box degli oggetti rilevati
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path=f"{Path}yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt", force_reload=True)
    # model = attempt_load(weights=f"{Path}yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt")
    
    # Esegui l'inferenza sull'immagine
    results = detect_image(model, frame)
    # results = model(frame)
    detections = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)
    print(detections, 'ciao')
    for result in detections:
        # Ottieni le coordinate del bounding box
        x, y, w, h = result[0:4]

        # Ottieni la classe associata
        class_index = int(result[5])
        class_name = model.names[class_index]

        # Fai qualcosa con le informazioni ottenute
        print("Classe:", class_name)
        print("Coordinate bounding box:", (x, y, w, h))

    # Traccia gli oggetti utilizzando DeepSORT
    tracked_objects = run(detection_file=detections, sequence_dir=frame, min_confidence=0.3, nn_budget=100, display=True)

    return tracked_objects

# Funzione per rilevare un tiro verso il canestro
def detect_shot(tracked_objects):
    for obj in tracked_objects:
        # Controllo se l'oggetto è una palla da basket
        if obj.class_name == 'basketball':
            # Esegue il controllo per determinare se viene effettuato un tiro
            if obj.area > ThresholdDirection and obj.velocity > ThresholdVelocity:
                return True
    return False

# Loop principale del video
cap = cv2.VideoCapture(f'{Path}dataset/ours/video/video1.mp4')  # Inserisci il percorso del video

while cap.isOpened():
    
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    if not ret:
        break

    # Rileva e traccia gli oggetti nel frame
    tracked_objects = track_objects(frame)

    # Rileva un tiro verso il canestro
    shot_detected = detect_shot(tracked_objects)

    # Visualizza il frame con le informazioni di tracciamento
    for obj in tracked_objects:
        cv2.rectangle(frame, (obj.left, obj.top), (obj.right, obj.bottom), (0, 255, 0), 2)
        cv2.putText(frame, obj.class_name, (obj.left, obj.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if shot_detected:
        cv2.putText(frame, "Tiro verso il canestro!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

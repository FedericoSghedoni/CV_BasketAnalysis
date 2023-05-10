import cv2
import numpy as np
# Carica il modello YOLO addestrato
net = cv2.dnn.readNet("yolov5/runs/train/yolo_basket_det_PDataset/weights/best.pt", "yolov5/models/yolov5s.yaml", )

# Carica i nomi delle classi
classes = []
with open("dataset/data.yaml", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Inizializza il video stream dal tuo telefono
cap = cv2.VideoCapture('http://10.80.154.236:8080/video')

# Loop per elaborare i frame del video stream
while True:
    # Leggi il frame successivo dal video stream
    ret, frame = cap.read()

    # Esegui l'inferenza degli oggetti sul frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Analizza le informazioni sugli oggetti rilevati
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                # Disegna il rettangolo intorno all'oggetto rilevato
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype("int")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[classId], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostra il frame elaborato
    cv2.imshow("YOLO Object Detection", frame)
    key = cv2.waitKey(1)

    # Esci dal loop se viene premuto il tasto "q"
    if key == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from tokenizer import Tokenizer

Path = 'distance/'
model_path = 'yolov8s_custom/weights/best.pt'

# Dimensioni dell'oggetto di riferimento noto (ad es. una scheda) in millimetri
oggetto_di_riferimento_larghezza_mm = 450
oggetto_di_riferimento_altezza_mm = 500

# Carica il video
cap = cv2.VideoCapture(f'{Path}IMG_4008.mp4')

tokenizer = Tokenizer(model_path)

ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
height, width, channels = frame.shape
video_writer = cv2.VideoWriter(f'{Path}IMG_4008b.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rileva e traccia gli oggetti nel frame
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    new_frame, detection = tokenizer.detect_objects(frame=frame)
    for result in detection:
        # Get the object class
        class_index = int(result.cls.item())

    video_writer.write(new_frame)

'''   
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # Calcola la larghezza dell'oggetto in pixel
            larghezza_oggetto_pixel = w

            # Calcola la distanza reale utilizzando le dimensioni dell'oggetto di riferimento noto
            distanza_reale_mm = (oggetto_di_riferimento_larghezza_mm * larghezza_oggetto_pixel) / width

            label = f"{classes[class_ids[i]]}: {distanza_reale_mm:.2f} mm"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''  




video_writer.release()
cap.release()
cv2.destroyAllWindows()

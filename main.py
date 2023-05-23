import cv2
import torch
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort

Path = 'C:/Users/Computer/Documents/GitHub/CV_BasketAnalysis/'

def detect_objects(frame, model, tracker):
    detections = model(frame)

    for result in detections.xywh:
        # Ottieni le coordinate del bounding box

        for i in range(result.shape[0]):
            # Ottieni la classe associata
            class_index = int(result[i,5])
            class_name = model.names[class_index]

            if class_name == 'basketball':
                track_objects(frame, result[i], tracker)
            
            # Visualizza classe e coordinate delle bounding box
            # print("Classe:", class_name)
            # print("Coordinate bounding box:", result[i,0:4])

    return detections

def track_objects(frame, detections, tracker):
    # detections expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    print(detections)
    print((np.array(detections[0:4]).tolist(),detections[4].item(),detections[5].item()))
    tracks = tracker.update_tracks((detections), frame=frame) 
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        print(track_id, ltrb)

# Loop principale del video
cap = cv2.VideoCapture(f'{Path}dataset/ours/video/video3.mp4')
model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path=f"{Path}yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt", force_reload=True)
tracker = DeepSort(max_age=5)
while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret:
        break

    # Rileva e traccia gli oggetti nel frame
    results = detect_objects(frame, model, tracker)
    break

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
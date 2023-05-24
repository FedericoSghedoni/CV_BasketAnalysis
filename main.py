import cv2
import torch
import numpy as np

from filterpy.kalman import KalmanFilter
# from deep_sort_realtime.deepsort_tracker import DeepSort

Path = 'C:/Users/Computer/Documents/GitHub/CV_BasketAnalysis/'

def detect_objects(frame, model, kalmanFilter):
    detections = model(frame)
    # To show the detection use the line below
    # detections.show()
    detections.save()
    for result in detections.xywh:
        # Ottieni le coordinate del bounding box

        for i in range(result.shape[0]):
            # Ottieni la classe associata
            class_index = int(result[i,5])
            class_name = model.names[class_index]

            # Se l'oggetto viene identificato come una basketball effettua la predizione della sua traiettoria
            if class_name == 'basketball' and result[i,4] >= 0.4:
                filtered_x, filtered_y = track_objects(result[i], kalmanFilter)
                cv2.circle(frame, (int(filtered_x), int(filtered_y)), 5, (0, 255, 0), -1)
            # Visualizza classe e coordinate delle bounding box
            # print("Classe:", class_name)
            # print("Coordinate bounding box:", result[i,0:4])

    return frame

def track_objects(detections, kalman_filter):
    x,y,w,h = detections[0:4]
    x,y,w,h = x.item(),y.item(),w.item(),h.item()
    kalman_filter.x = np.array([x, y, 0, 0]).reshape(4, 1)
    # detections expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    # Genera la nuova misurazione della posizione della palla (simulata)
    z = np.array([[x + np.random.randn(1)[0] * 2],
                    [y + np.random.randn(1)[0] * 2]])

    # Previsione del filtro di Kalman
    kalman_filter.predict()

    # Aggiornamento del filtro di Kalman con la nuova misurazione
    kalman_filter.update(z)

    # Recupero della posizione filtrata della palla
    filtered_state = kalman_filter.x
    filtered_x = filtered_state[0][0]
    filtered_y = filtered_state[1][0]

    return filtered_x, filtered_y

# Loop principale del video
model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path=f"{Path}yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt", force_reload=True)

kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
kalman_filter.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
kalman_filter.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
kalman_filter.P *= 1000
kalman_filter.R = np.array([[0.1, 0],
                            [0, 0.1]])
kalman_filter.Q = np.array([[0.1, 0, 0, 0],
                            [0, 0.1, 0, 0],
                            [0, 0, 0.1, 0],
                            [0, 0, 0, 0.1]])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
frame_width, frame_height = 640, 480
video_writer = cv2.VideoWriter('video_detections.mp4', fourcc, fps, (frame_width, frame_height))

cap = cv2.VideoCapture(f'{Path}dataset/ours/video/video3.mp4')
while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret:
        break

    # Rileva e traccia gli oggetti nel frame
    new_frame = detect_objects(frame, model, kalman_filter)
    video_writer.write(frame)

video_writer.release()
cap.release()
cv2.destroyAllWindows()
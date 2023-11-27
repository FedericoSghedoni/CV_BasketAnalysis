import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import sys

# !!! THIS VERSION SHOW YOU THE IMG BEFORE SAVING IT

# Get the video files in the folder
folder_path= '../CVDataset/dataset/ours/'
model = YOLO('yolov8s_final/weights/best.pt')
classes = torch.Tensor([0., 1., 2.])
folder = 'dataset2/'
# Crea le cartelle di destinazione se non esistono già
for f in ['', 'detected_files/', 'new_labels/', 'new_images/']:
    if not os.path.exists(os.path.join(folder, f)):
        os.makedirs(os.path.join(folder, f))
j = 0 # number of opened files, used for naming the new files

for video_file in [f for f in os.listdir(folder_path) if f.endswith('.MOV')]:
    j += 1
    video_path = os.path.join(folder_path, video_file)
    size = len(video_file)
    name = video_file[:size - 4]
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue

    # Read and process frames one at a time
    i = 0 # number of frames seen, used for naming the new files
    k = 5 # number of frames since last saved frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of the video file
        
        # Ritagliare l'immagine: taglia 190 pixel dall'alto e 370 dal basso
        frame = frame[190:frame.shape[:2][0]-370, :]
        i += 1 # add 1 to the count
        k += 1 # add 1 to the count
        if k >= 20:
            results = model(frame, conf=0.25, verbose=True)
            for r in results:
                if set(r.boxes.cls.tolist()) == set(classes.tolist()):
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    # Inizializza una finestra OpenCV per mostrare le immagini
                    cv2.namedWindow('Immagini', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Immagini', im_array)
                    print(f"Salvare il file? (0 = Sì, 1 = No) o ESC per uscire: ")
                    k = cv2.waitKey(0)
                    #print(k)
                    scelta = k - 48
                    print(scelta)
                    if scelta == 0:
                        print(f'{folder}detected_files/{name}frame{i}.jpg')
                        print(f'{folder}new_labels/{name}frame{i}.txt')
                        r.save_txt(f'{folder}new_labels/{name}frame{i}.txt')
                        cv2.imwrite(f'{folder}new_images/{name}frame{i}.jpg', frame)
                        im.save(f'{folder}detected_files/{name}frame{i}.jpg')  # save image
                        k = 0
                    elif scelta == 1:
                        pass
                    elif scelta == -21:
                        sys.exit()


cap.release()
cv2.destroyAllWindows()
print(j)
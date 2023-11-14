import os
import cv2
import torch

from PIL import Image
from ultralytics import YOLO

# Get the video files in the folder
folder_path= '../CVDataset/dataset/ours/video3/'
model = YOLO('yolov8s_custom/weights/best.pt')
classes = torch.Tensor([1., 2., 3.])
folder = 'dataset5/'
# Crea le cartelle di destinazione se non esistono già
for f in ['', 'detected_files/', 'new_labels/', 'new_images/']:
    if not os.path.exists(os.path.join(folder, f)):
        os.makedirs(os.path.join(folder, f))
j = 0 # number of opened files, used for naming the new files

for video_file in [f for f in os.listdir(folder_path) if f.endswith('.mp4')]:
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
        
        i += 1 # add 1 to the count
        k += 1 # add 1 to the count
        if k >= 5:
            results = model(frame, conf=0.4)
            for r in results:
                if set(r.boxes.cls.tolist()) == set(classes.tolist()):
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    print(f'{folder}detected_files/{name}frame{i}.jpg')
                    print(f'{folder}new_labels/{name}frame{i}.txt')
                    r.save_txt(f'{folder}new_labels/{name}frame{i}.txt')
                    cv2.imwrite(f'{folder}new_images/{name}frame{i}.jpg', frame)
                    im.save(f'{folder}detected_files/{name}frame{i}.jpg')  # save image
                    k = 0

cap.release()
cv2.destroyAllWindows()
print(j)
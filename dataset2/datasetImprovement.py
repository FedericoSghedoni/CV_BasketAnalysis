import os
import cv2
import torch

from PIL import Image
from ultralytics import YOLO

# Get the video files in the folder
video_files = [f for f in os.listdir('dataset/ours/video') if f.endswith('.mp4')]
model = YOLO('yolov8s_custom/weights/best.pt')
classes = torch.Tensor([1., 2., 3.])
path = 'dataset2/'
j = 0 # number of opened files, used for naming the new files

for video_file in video_files:
    j += 1
    video_path = os.path.join('dataset/ours/video', video_file)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue

    # Read and process frames one at a time
    i = 0 # number of frames seen, used for naming the new files
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # End of the video file

        results = model(frame)

        for r in results:
            
            if set(r.boxes.cls.tolist()) == set(classes.tolist()):
                i += 1
                
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                print(f'{path}detected_files/video{j}frame{i}.jpg')
                print(f'{path}new_labels/video{j}frame{i}.txt')
                r.save_txt(f'{path}new_labels/video{j}frame{i}.txt')
                # cv2.imwrite(f'{path}new_images/video{j}frame{i}.jpg', frame)
                # im.save(f'{path}detected_files/video{j}frame{i}.jpg')  # save image
                break


cap.release()
cv2.destroyAllWindows()
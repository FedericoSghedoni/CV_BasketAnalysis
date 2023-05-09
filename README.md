# CV_BasketAnalysis
    git submodule init
    git submodule update
    requirements: pip install -r yolov5/requirements.txt
    
    train-yolo: python yolov5/train.py --img 416 --cfg yolov5/models/yolov5s.yaml --hyp yolov5/data/hyps/hyp.scratch-med.yaml --batch 14 --epochs 20 --data dataset/data.yaml --weights yolov5s.pt --workers 24 --name yolo_basket_det_PDataset --device 0

    weights path: yolov5/runs/train/yolo_basket_det
    
    inference-yolo: python yolov5/detect.py --source dataset/test/images --weights yolov5/runs/train/yolo_basket_det_PDataset/weights/best.pt --conf 0.25 --name yolo_basket_det_PDataset

    use with camera: python yolov5/detect.py --weights yolov5/runs/train/yolo_basket_det_PDataset/weights/best.pt --source 0
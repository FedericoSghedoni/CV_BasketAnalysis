# CV_BasketAnalysis
    git submodule init
    git submodule update
    git submodule update --remote
    requirements: pip install -r yolo5/requirements.txt
    
    train-yolo: python yolo5/train.py --img 416 --cfg yolo5/models/yolov5s.yaml --hyp yolo5/data/hyps/hyp.scratch-med.yaml --batch 14 --epochs 20 --data dataset/data.yaml --weights yolov5s.pt --workers 24 --name yolo_basket_det_PDataset --device 0

    weights path: yolo5/runs/train/yolo_basket_det
    
    inference-yolo: python yolo5/detect.py --source dataset/test/images --weights yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt --conf 0.25 --name yolo_basket_det_PDataset

    use with camera: python yolo5/detect.py --weights yolo5/runs/train/yolo_basket_det_PDataset/weights/best.pt --source 0
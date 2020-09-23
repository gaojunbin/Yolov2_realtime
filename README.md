# YOLOv2 real-time detect

2020.9

## What you need

1. Clone our project

   ```shell
   git clone https://github.com/gaojunbin/Yolov2_realtime
   ```

2. get the YOLOv2 MS COCO weights

   ```shell
   mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights
   ```

## Detect an image

```shell
python detect.py
```

## Real-time

```shell
python realtime_detect.py
```


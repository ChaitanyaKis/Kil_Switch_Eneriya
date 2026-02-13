from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, camera_id, timestamp):
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": results.names[cls_id],
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "camera_id": camera_id,
                "timestamp": timestamp
            })

        return detections

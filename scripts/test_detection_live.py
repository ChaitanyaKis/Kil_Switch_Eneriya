from pathlib import Path

from src.ingestion.video_loader import VideoLoader
from src.detection.yolo_detector import YOLODetector
from src.detection.object_filter import ObjectFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
video_path = PROJECT_ROOT / "data" / "test_clips" / "cam1.mp4"

loader = VideoLoader(str(video_path), camera_id="cam_1")
detector = YOLODetector()
filterer = ObjectFilter()

for cam_id, ts, frame in loader.frames():
    detections = detector.detect(frame, cam_id, ts)
    detections = filterer.filter(detections)

    if detections:
        print("\nTRACKABLE OBJECTS:")
        for d in detections:
            print(d)

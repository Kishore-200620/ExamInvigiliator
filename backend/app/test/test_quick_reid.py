import sys
sys.path.insert(0, r"D:\clg project\ExamInvigilationAI\backend")
from app.ai.detector.manager import DetectionManager
from app.ai.detector.yolo import YOLODetector
from app.ai.tracker.manager import TrackingManager
from app.ai.video.capture import OpenCVCapture
from app.ai.video.manager import VideoManager
import time
video = VideoManager(OpenCVCapture(r"D:\clg project\ExamInvigilationAI\sample_clips\clip2.mp4"), source_id='exam_video')
detector = DetectionManager(YOLODetector())
tracker = TrackingManager()
video.start()
detector.start()
DETECT_EVERY_N=2
frame_counter=0
last_detections=[]
for _ in range(12):
    frame=None
    for __ in range(30):
        frame=video.read()
        if frame is not None:
            break
        time.sleep(0.005)
    if frame is None:
        break
    frame_counter+=1
    should_detect=(frame_counter%DETECT_EVERY_N==0) or frame_counter==1
    if should_detect:
        t0=time.perf_counter()
        detections=detector.detect(frame)
        dt=(time.perf_counter()-t0)*1000
        tracks=tracker.update(detections)
        last_detections=detections
        c=sum(1 for t in tracks if t.state.name=="CONFIRMED")
        print(f"f{frame.frame_id} DETECT dt={dt:.0f}ms dets={len(detections)} tracks={len(tracks)} C={c}")
    else:
        for t in tracker.tracks:
            t.predict()
        tracks=tracker.tracks
        print(f"f{frame.frame_id} SKIP dets={len(last_detections)} tracks={len(tracks)}")
video.stop()
detector.stop()
print('done yolo26n 1280 conf 0.15')

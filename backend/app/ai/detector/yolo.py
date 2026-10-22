from ultralytics import YOLO
import time
import numpy as np
from app.ai.detector.base import Detector
from app.ai.detector.models import BoundingBox, Detection
from app.ai.video.frame import Frame
from app.config.settings import settings
from app.ai.detector.metrics import DetectorMetrics


class YOLODetector(Detector):
    """
    YOLO26l implementation.
    """

    PERSON_CLASS = 0

    def __init__(self):
        self.model = None
        self.metrics = DetectorMetrics()

    def load_model(self) -> None:
        # Resolve to absolute path so ultralytics doesn't resolve "yolo26n-pose" to the base yolo26n.pt (COCO only, no pose)
        import pathlib
        model_path = pathlib.Path(settings.YOLO_MODEL)
        if not model_path.is_absolute():
            # Resolve relative to project root (where settings file lives)
            proj_root = pathlib.Path(__file__).resolve().parents[2]  # backend/app/ai/detector -> backend -> project
            model_path = proj_root / settings.YOLO_MODEL
        # If relative path includes backend/, resolve properly
        if not model_path.exists():
            # The settings path may point to workspace root; try resolving relative to backend/models
            backend_models = pathlib.Path(__file__).resolve().parents[3] / "models" / pathlib.Path(settings.YOLO_MODEL).name
            if backend_models.exists():
                model_path = backend_models.resolve()
            else:
                alt = pathlib.Path(settings.YOLO_MODEL)
                if alt.exists():
                    model_path = alt.resolve()
        # Ensure absolute for ultralytics
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        self.model = YOLO(str(model_path))

    def detect(self, frame: Frame) -> list[Detection]:

        if self.model is None:
            raise RuntimeError("YOLO model not loaded.")
        start = time.perf_counter()

        results = self.model.predict(
            frame.image,
            imgsz=settings.YOLO_IMAGE_SIZE,
            conf=settings.YOLO_CONFIDENCE,
            iou=settings.YOLO_IOU,
            device=settings.DEVICE,
            verbose=settings.YOLO_VERBOSE,
        )
        inference_time = (
            time.perf_counter() - start
            ) * 1000

        detections: list[Detection] = []
        # For diagnostic: count raw boxes before class filtering (all classes)
        raw_count = 0
        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                try:
                    raw_count += len(result.boxes)
                except Exception:
                    raw_count += 0

        for result in results:
            # YOLO pose: boxes and keypoints are 1:1 aligned by order
            # We use enumerate to get the exact box index within this result
            for box_idx, box in enumerate(result.boxes):

                class_id = int(box.cls.item())

                if class_id != self.PERSON_CLASS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # If YOLO pose model is used, extract keypoints for this detection
                kpts = None
                if hasattr(result, "keypoints") and result.keypoints is not None and result.keypoints.data is not None and len(result.keypoints.data) > 0:
                    try:
                        # box_idx from enumerate is the exact index in this result
                        if box_idx < len(result.keypoints.data):
                            kp_data = result.keypoints.data[box_idx]
                            # kp_data shape (17, 3) x,y,conf
                            if hasattr(kp_data, "cpu"):
                                kp_data = kp_data.cpu().numpy()
                            kpts = np.array(kp_data)  # (17,3)
                        else:
                            # Fallback: no matching keypoint set for this box index
                            kpts = None

                    except Exception:
                        # Keypoints unavailable for this detection (e.g., occluded face)
                        kpts = None

                detection = Detection(
                    frame_id=frame.frame_id,
                    bbox=BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=result.names[class_id],
                    metadata={"keypoints": kpts} if kpts is not None else {},
                )

                detections.append(detection)

        # Diagnostic logging for YOLO stage (only when enabled, no overhead otherwise)
        try:
            from app.core.diagnostic import diagnostic_log, is_diagnostic_frame

            if is_diagnostic_frame(frame.frame_id):
                h, w = frame.image.shape[:2]
                # Frame info will be logged by tracker, but also log detection-specific
                lines = []
                lines.append("FRAME")
                lines.append(f"frame_id: {frame.frame_id}")
                lines.append(f"timestamp: {frame.timestamp}")
                lines.append(f"image_width: {w}")
                lines.append(f"image_height: {h}")
                lines.append(f"raw_detection_count: {raw_count}")
                lines.append(f"final_detection_count: {len(detections)}")
                # Note: raw includes all classes before PERSON filter; ultralytics NMS already applied inside predict
                # Filtering by class PERSON is the only post-NMS filter we do
                if raw_count != len(detections):
                    lines.append(f"filtered_by_class: {raw_count - len(detections)} (non-person)")
                diagnostic_log(frame.frame_id, "\n".join(lines))
                # Per-detection details
                for idx, det in enumerate(detections):
                    bb = det.bbox
                    diagnostic_log(
                        frame.frame_id,
                        "\n".join(
                            [
                                "DETECTION",
                                f"frame_id: {frame.frame_id}",
                                f"detection_index: {idx}",
                                f"bbox: {bb.x1},{bb.y1},{bb.x2},{bb.y2}",
                                f"width: {bb.width}",
                                f"height: {bb.height}",
                                f"confidence: {det.confidence:.3f}",
                                f"class: {det.class_id} ({det.class_name})",
                            ]
                        ),
                    )
                # Warnings for duplicate / merged detections
                # Warning 2: possible duplicate detections (high IoU between two detections)
                try:
                    from app.ai.tracker.cost import CostCalculator

                    calc = CostCalculator()
                    for i in range(len(detections)):
                        for j in range(i + 1, len(detections)):
                            iou = calc.iou(detections[i].bbox, detections[j].bbox)
                            if iou > 0.7:  # very high overlap -> likely duplicate
                                diagnostic_log(
                                    frame.frame_id,
                                    "\n".join(
                                        [
                                            "WARNING POSSIBLE_DUPLICATE_DETECTION",
                                            f"frame_id: {frame.frame_id}",
                                            f"detection_a: {i} bbox {detections[i].bbox.x1},{detections[i].bbox.y1},{detections[i].bbox.x2},{detections[i].bbox.y2} conf {detections[i].confidence:.2f}",
                                            f"detection_b: {j} bbox {detections[j].bbox.x1},{detections[j].bbox.y1},{detections[j].bbox.x2},{detections[j].bbox.y2} conf {detections[j].confidence:.2f}",
                                            f"IoU: {iou:.3f}",
                                            f"confidence_a: {detections[i].confidence:.3f}",
                                            f"confidence_b: {detections[j].confidence:.3f}",
                                        ]
                                    ),
                                )
                except Exception:
                    pass
                # Warning 1: possible merged-person (unusually wide bbox)
                for idx, det in enumerate(detections):
                    bb = det.bbox
                    # Heuristic: width > 1.5*height and width > 200px in 1536 image may contain two people
                    if bb.width > 200 and bb.width > 1.4 * bb.height:
                        diagnostic_log(
                            frame.frame_id,
                            "\n".join(
                                [
                                    "WARNING POSSIBLE_MERGED_PERSON",
                                    f"frame_id: {frame.frame_id}",
                                    f"detection_index: {idx}",
                                    f"bbox: {bb.x1},{bb.y1},{bb.x2},{bb.y2}",
                                    f"width: {bb.width}",
                                    f"height: {bb.height}",
                                ]
                            ),
                        )
        except Exception:
            pass
        self.metrics.total_frames += 1

        self.metrics.detections = len(detections)

        self.metrics.inference_time_ms = inference_time

        self.metrics.processing_fps = (
            1000.0 / inference_time
            if inference_time > 0
            else 0.0
        )

        self.metrics.average_inference_ms = (
            (
                self.metrics.average_inference_ms
                * (self.metrics.total_frames - 1)
                + inference_time
            )
            / self.metrics.total_frames
        )

        self.metrics.average_fps = (
            (
                self.metrics.average_fps
                * (self.metrics.total_frames - 1)
                + self.metrics.processing_fps
            )
            / self.metrics.total_frames
        )
        return detections

    def detect_batch(self, frames: list[Frame]) -> list[list[Detection]]:
        """Infer two (or fewer) frames together, returning detections in input order."""
        if not frames:
            return []
        if self.model is None:
            raise RuntimeError("YOLO model not loaded.")
        start = time.perf_counter()
        results = self.model.predict(
            [frame.image for frame in frames],
            imgsz=settings.YOLO_IMAGE_SIZE,
            conf=settings.YOLO_CONFIDENCE,
            iou=settings.YOLO_IOU,
            device=settings.DEVICE,
            verbose=settings.YOLO_VERBOSE,
            batch=len(frames),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        batches: list[list[Detection]] = []
        for frame_index, frame in enumerate(frames):
            if frame_index >= len(results):
                batches.append([])
                continue
            result = results[frame_index]
            detections: list[Detection] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box_idx, box in enumerate(boxes):
                    class_id = int(box.cls.item())
                    if class_id != self.PERSON_CLASS:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    keypoints = None
                    pose = getattr(result, "keypoints", None)
                    if pose is not None and getattr(pose, "data", None) is not None and box_idx < len(pose.data):
                        try:
                            kp_data = pose.data[box_idx]
                            if hasattr(kp_data, "cpu"):
                                kp_data = kp_data.cpu().numpy()
                            keypoints = np.array(kp_data)
                        except Exception:
                            keypoints = None
                    detections.append(Detection(
                        frame_id=frame.frame_id,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=float(box.conf.item()),
                        class_id=class_id,
                        class_name=result.names[class_id],
                        metadata={"keypoints": keypoints} if keypoints is not None else {},
                    ))
            batches.append(detections)

        per_frame_ms = elapsed_ms / len(frames)
        self.metrics.total_frames += len(frames)
        self.metrics.detections = len(batches[-1]) if batches else 0
        self.metrics.inference_time_ms = per_frame_ms
        self.metrics.processing_fps = 1000.0 / per_frame_ms if per_frame_ms > 0 else 0.0
        previous_total = self.metrics.total_frames - len(frames)
        self.metrics.average_inference_ms = (
            (self.metrics.average_inference_ms * previous_total + elapsed_ms) / self.metrics.total_frames
            if self.metrics.total_frames else per_frame_ms
        )
        return batches
    def get_metrics(self) -> DetectorMetrics:
        return self.metrics
    
    def unload_model(self) -> None:
        self.model = None

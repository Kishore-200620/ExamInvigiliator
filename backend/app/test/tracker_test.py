import time
import cv2

from app.ai.detector.manager import DetectionManager
from app.ai.detector.yolo import YOLODetector
from app.ai.tracker.manager import TrackingManager
from app.ai.video.capture import OpenCVCapture
from app.ai.video.manager import VideoManager


VIDEO_PATH = r"D:\clg project\ExamInvigilationAI\sample_clips\clip5.mp4"


def main():

    # ----------------------------
    # Video
    # ----------------------------
    video = VideoManager(
        OpenCVCapture(VIDEO_PATH),
        source_id="exam_video",
    )

    # ----------------------------
    # Detector
    # ----------------------------
    detector = DetectionManager(
        YOLODetector(),
    )

    # ----------------------------
    # Tracker
    # ----------------------------
    tracker = TrackingManager()

    if not video.start():
        print("Unable to start video.")
        return

    detector.start()

    cv2.namedWindow(
        "Tracker Debug",
        cv2.WINDOW_NORMAL,
    )

    last_print = time.time()
    frame_counter = 0

    # Smooth playback: detect every N frames, reuse predicted tracks otherwise
    # yolo26n @640 on CPU ~80-150ms, so N=2-3 keeps ~15-20 FPS display without stutter
    DETECT_EVERY_N = 2
    last_detections = []
    last_tracks = []
    # For FPS smoothing of display loop (independent of capture FPS)
    display_fps_ema = 30.0
    last_display_time = time.perf_counter()

    print("=" * 70)
    print("TRACKER DEBUG STARTED (smooth mode: yolo26n 1536, detect every 2 frames, queue 30)")
    print(f"Video: {VIDEO_PATH}")
    print("Press Q to Exit")
    print("=" * 70)

    try:

        while video.is_running():

            frame = video.read()

            if frame is None:
                # No frame available yet - small sleep to avoid busy spin, keeps display smooth
                time.sleep(0.005)
                # Still check if video finished (no more frames coming)
                if not video.is_running():
                    break
                continue

            frame_counter += 1

            # Smooth display FPS calc (independent of capture fps)
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, now - last_display_time)
            display_fps_ema = 0.85 * display_fps_ema + 0.15 * inst_fps
            last_display_time = now

            # Run detector only every N frames - huge CPU saving, prevents queue stall
            # Pass frame for ReID + head pose event-triggered (CONFIRMED+30/45, not every track every frame)
            should_detect = (frame_counter % DETECT_EVERY_N == 0) or (frame_counter == 1)
            if should_detect:
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame=frame)
                last_detections = detections
                last_tracks = tracks
            else:
                # Predict-only for intermediate frames - keeps boxes moving smoothly via Kalman
                # Don't call tracker.update (would create misses); just predict existing tracks
                for t in tracker.tracks:
                    t.predict()
                detections = last_detections
                tracks = tracker.tracks
                # Still update hits/misses aging via predict above; last_tracks updated
                last_tracks = tracks

            confirmed = 0
            tentative = 0
            lost = 0

            # -------------------------------------------------
            # Draw Tracks + YOLO Pose dots (only dots, till wrist + eyes/ears, high visibility)
            # -------------------------------------------------

            # First draw YOLO pose dots from detections (the actual YOLO pose, not second model)
            for det in detections:
                kpts = det.metadata.get("keypoints") if hasattr(det, "metadata") and isinstance(det.metadata, dict) else None
                if kpts is not None:
                    # Draw dots till wrist + eyes/ears per your ask (no hips/knees/ankles)
                    for idx in [0,1,2,3,4,5,6,7,8,9,10]:  # nose, eyes, ears, shoulders, elbows, wrists
                        if idx < len(kpts):
                            x,y,c = kpts[idx]
                            if c > 0.3:
                                # High visibility: white border 7px + orange inner 5px
                                cv2.circle(frame.image, (int(x),int(y)), 5, (255,255,255), -1)
                                cv2.circle(frame.image, (int(x),int(y)), 4, (0,165,255), -1)

            for track in tracks:

                if track.state.name == "CONFIRMED":
                    confirmed += 1
                    color = (0, 255, 0)

                elif track.state.name == "TENTATIVE":
                    tentative += 1
                    color = (0, 255, 255)

                else:
                    lost += 1
                    color = (0, 0, 255)

                box = track.motion.bbox

                cv2.rectangle(
                    frame.image,
                    (box.x1, box.y1),
                    (box.x2, box.y2),
                    color,
                    2,
                )

                cv2.putText(
                    frame.image,
                    f"ID:{track.track_id}",
                    (box.x1, box.y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                cv2.putText(
                    frame.image,
                    f"H:{track.statistics.hits} M:{track.statistics.misses}",
                    (box.x1, box.y1 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

                # Phase 5: head pose visualization (yaw/pitch) if available in pose_history
                if hasattr(track, "pose_history") and track.pose_history:
                    pose = track.pose_history[-1]
                    # pose is dict {yaw,pitch,roll,confidence} or HeadPose object
                    try:
                        yaw = pose.get("yaw", 0) if isinstance(pose, dict) else getattr(pose, "yaw", 0)
                        pitch = pose.get("pitch", 0) if isinstance(pose, dict) else getattr(pose, "pitch", 0)
                        # Draw yaw/pitch near bbox
                        cv2.putText(
                            frame.image,
                            f"Y:{yaw:.0f} P:{pitch:.0f}",
                            (box.x1, box.y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 0),
                            1,
                        )
                        # Draw gaze arrow from head center
                        cx = (box.x1 + box.x2)//2
                        cy = box.y1 + int((box.y2 - box.y1)*0.2)  # head center upper 40%
                        # Simple yaw arrow: length 20 * sin(yaw)
                        import math
                        arrow_len = 20
                        dx = int(arrow_len * math.sin(math.radians(yaw)))
                        dy = int(arrow_len * -math.sin(math.radians(pitch)))
                        cv2.arrowedLine(frame.image, (cx, cy), (cx+dx, cy+dy), (255,255,0), 1, tipLength=0.3)
                    except Exception:
                        pass
                # Phase 4: draw small dot if ReID embedding exists (feature_history) - increased 3->5px
                if hasattr(track, "feature_history") and track.feature_history:
                    # Green dot at top-right of bbox
                    cv2.circle(frame.image, (box.x2 - 5, box.y1 + 5), 5, (255,0,0), -1)

            # -------------------------------------------------
            # Debug Panel
            # -------------------------------------------------

            cv2.rectangle(
                frame.image,
                (10, 10),
                (360, 210),
                (35, 35, 35),
                -1,
            )

            infer_ms = detector.detector.metrics.inference_time_ms if hasattr(detector.detector, 'metrics') else 0
            # Phase 5 pose stats for visual
            with_pose = sum(1 for t in tracks if hasattr(t, "pose_history") and t.pose_history)
            with_reid = sum(1 for t in tracks if hasattr(t, "feature_history") and t.feature_history)
            debug_info = [

                f"FPS video    : {frame.fps:.1f}",
                f"FPS display  : {display_fps_ema:.1f}",
                f"Infer      : {infer_ms:.0f}ms",
                f"Q            : {video.qsize()}",
                f"Frame        : {frame.frame_id}",
                "",
                f"Detections   : {len(detections)} {'(cached)' if not should_detect else ''}",
                f"Tracks       : {len(tracks)}",
                "",
                f"Confirmed    : {confirmed}",
                f"Tentative    : {tentative}",
                f"Lost         : {lost}",
                f"Pose         : {with_pose}/{len(tracks)}",
                f"ReID         : {with_reid}/{len(tracks)}",

            ]

            y = 35

            for text in debug_info:

                cv2.putText(
                    frame.image,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                y += 22

            # -------------------------------------------------
            # Console Debug
            # -------------------------------------------------

            if time.time() - last_print >= 1:

                avg_hits = (
                    sum(t.statistics.hits for t in tracks)
                    / max(1, len(tracks))
                )

                avg_misses = (
                    sum(t.statistics.misses for t in tracks)
                    / max(1, len(tracks))
                )

                duplicate_ids = len(
                    set(t.track_id for t in tracks)
                ) != len(tracks)

                # Relaxed leak threshold: allow +20 or 1.5x detections for crowded 60-person scenes
                track_leak = len(tracks) > max(len(detections) + 20, int(len(detections) * 1.5))

                # Rescued stats per 100 frames
                rescued_window = tracker.window_rescued
                rescued_total = tracker.total_rescued
                rescued_ious = getattr(tracker, "_rescued_ious_window", [])
                if rescued_ious:
                    avg_rescue_iou = sum(rescued_ious) / len(rescued_ious)
                    min_iou = min(rescued_ious)
                    max_iou = max(rescued_ious)
                    iou_dist = f"{min_iou:.2f}-{max_iou:.2f} avg {avg_rescue_iou:.2f} n={len(rescued_ious)}"
                else:
                    iou_dist = "n=0"

                print("=" * 70)

                print(f"Frame              : {frame.frame_id}")
                print(f"FPS video          : {frame.fps:.1f}")
                print(f"FPS display        : {display_fps_ema:.1f}")
                print(f"Infer ms           : {detector.detector.metrics.inference_time_ms:.1f}" if hasattr(detector.detector, 'metrics') else "Infer ms           : N/A")

                print()

                print(f"Detections         : {len(detections)}")
                print(f"Tracks             : {len(tracks)}")

                print()

                print(f"Confirmed          : {confirmed}")
                print(f"Tentative          : {tentative}")
                print(f"Lost               : {lost}")

                print()

                print(f"Average Hits       : {avg_hits:.2f}")
                print(f"Average Misses     : {avg_misses:.2f}")

                print()

                # Phase 5/4 stats
                with_pose = sum(1 for t in tracks if hasattr(t, "pose_history") and t.pose_history)
                with_reid = sum(1 for t in tracks if hasattr(t, "feature_history") and t.feature_history)
                avg_pose = sum(len(getattr(t, "pose_history", [])) for t in tracks)/max(1,len(tracks))
                print(f"With Pose          : {with_pose}/{len(tracks)} avg {avg_pose:.1f}")
                print(f"With ReID          : {with_reid}/{len(tracks)}")
                print(f"Rescued (window)   : {rescued_window} total {rescued_total} IoUs {iou_dist}")

                print(
                    f"Track Leak         : {'YES' if track_leak else 'NO'}"
                )

                print(
                    f"Duplicate IDs      : {'YES' if duplicate_ids else 'NO'}"
                )

                print("=" * 70)

                # Clear per-window IoU log after reporting
                if hasattr(tracker, "_rescued_ious_window"):
                    tracker._rescued_ious_window = []

                last_print = time.time()

            cv2.imshow(
                "Tracker Debug",
                frame.image,
            )

            # ~1ms wait for smooth 30fps display; 1ms not 0ms prevents 100% CPU spin
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Optional throttle to cap display at ~30fps if detector faster than display
            # time.sleep(max(0, 0.033 - (time.perf_counter() - now)))

    finally:

        detector.stop()
        video.stop()

        cv2.destroyAllWindows()

        print("Tracker Debug Finished")


if __name__ == "__main__":
    main()
import argparse
import hashlib
import json
import math
import cv2
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YUNET_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2026may.onnx"
YUNET_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"

def get_video_info(video_path):
    vid = cv2.VideoCapture(str(video_path))

    if not vid.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid.release()

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video FPS")
    duration = frame_count / fps

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }

def iter_sampled_frames(video_path, interval=1):
    """Yield (source index, estimated seconds, BGR frame), without storing the video.

    Times use frame_index / reported FPS, not container presentation timestamps.
    Sampling matches the original extract_frames implementation.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Sampling interval must be finite and positive")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS")

    sample_count = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_index / fps
            if timestamp >= sample_count * interval:
                yield frame_index, timestamp, frame
                sample_count += 1
            frame_index += 1
    finally:
        cap.release()


def extract_frames(video_path, interval=1):
    """Legacy list-returning helper; process_video uses the streaming iterator."""
    return [frame for _, _, frame in iter_sampled_frames(video_path, interval)]

def detect_faces(frame):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_detector = cv2.CascadeClassifier(cascade_path)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces

def clip_face_box(frame, face_box):
    """Convert float (x, y, width, height) to a valid integer crop box or None."""
    x, y, w, h = map(float, face_box)
    if not all(math.isfinite(v) for v in (x, y, w, h, x + w, y + h)):
        return None
    if w <= 0 or h <= 0:
        return None
    height, width = frame.shape[:2]
    x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
    x1, y1 = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def crop_face(frame, face_box):
    box = clip_face_box(frame, face_box)
    if box is None:
        raise ValueError("Face box does not contain a valid image region")
    x, y, w, h = box
    return frame[y:y+h, x:x+w]

def preprocess_face(face_crop):
    if face_crop is None or face_crop.size == 0:
        raise ValueError("Cannot preprocess an empty face crop")
    resized_face = cv2.resize(face_crop, (224, 224))
    rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    normalized_face = rgb_face.astype("float32") / 255.0

    return normalized_face

def save_face(face, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_uint8 = (face * 255).round().clip(0, 255).astype("uint8")
    face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(output_path), face_bgr):
        raise OSError(f"Could not save face: {output_path}")

def detect_faces_mediapipe(frame, detector):
    """Retained optional baseline; MediaPipe is not required for the YuNet path."""
    import mediapipe as mp

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    result = detector.detect(mp_image)

    faces = []

    for detection in result.detections:
        box = detection.bounding_box
        faces.append((
            box.origin_x,
            box.origin_y,
            box.width,
            box.height
        ))

    return faces

def create_yunet_detector(model_path=DEFAULT_YUNET_MODEL, score_threshold=0.85):
    """Load once; detect_faces_yunet updates input dimensions for each frame."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")
    if not 0 < score_threshold <= 1:
        raise ValueError("Score threshold must be in (0, 1]")
    if model_path.resolve() == DEFAULT_YUNET_MODEL.resolve():
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != YUNET_SHA256:
            raise ValueError("Bundled YuNet model checksum mismatch")
    return cv2.FaceDetectorYN.create(
        str(model_path), "", (320, 320), score_threshold, 0.3, 5000
    )


def detect_faces_yunet(frame, detector):
    """Return float xywh boxes, uncalibrated detection scores and five landmarks.

    Input is a full OpenCV BGR frame. Detection order is not a track identity.
    """
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [
        {
            "bbox_xywh": [float(v) for v in face[:4]],
            "confidence": float(face[-1]),
            "landmarks_xy": face[4:14].reshape(5, 2).tolist(),
        }
        for face in faces
    ]


def process_video(video_path, output_dir, detector, interval=1, model_path=None):
    """Save full-frame YuNet crops and manifest.json; return saved crop count.

    This replaces the historical split-frame MediaPipe pipeline. A supplied
    detector must be a FaceDetectorYN instance, not a MediaPipe detector.
    Output must be new or empty, protecting existing results from overwrites.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Sampling interval must be finite and positive")
    video_info = get_video_info(video_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "video_path": str(Path(video_path).resolve()),
        "video_info": video_info,
        "detector": "YuNet",
        "opencv_version": cv2.__version__,
        "score_threshold": float(detector.getScoreThreshold()),
        "nms_threshold": float(detector.getNMSThreshold()),
        "top_k": int(detector.getTopK()),
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else None,
        "sampling_interval_seconds": interval,
        "timestamp_method": "source frame index / reported FPS; estimated, not presentation timestamps",
        "crop_format": "224x224 PNG; RGB float32 [0,1] before saving; model-specific normalization not applied",
        "limitations": ["Detection score is not a deepfake probability.",
                        "Detection index is not a facial track ID.",
                        "Only sampled frames are analyzed; partial or occluded faces may be missed."],
        "frames": [],
    }
    total_faces = 0
    sampled = iter_sampled_frames(video_path, interval)
    try:
        for frame_index, timestamp, frame in sampled:
            detections = detect_faces_yunet(frame, detector)
            frame_result = {"frame_index": frame_index, "timestamp_seconds_estimated": timestamp,
                            "detections": detections}
            for j, detection in enumerate(detections):
                box = clip_face_box(frame, detection["bbox_xywh"])
                detection["crop_bbox_xywh"] = box
                detection["crop_path"] = None
                if box is None:
                    detection["crop_status"] = "rejected_invalid_box"
                    continue
                processed_face = preprocess_face(crop_face(frame, box))
                filename = f"frame_{frame_index:06d}_face_{j}.png"
                save_face(processed_face, output_dir / filename)
                detection["crop_path"] = filename
                detection["crop_status"] = "saved"
                total_faces += 1
            frame_result["status"] = (
                "no_faces" if not detections else
                "faces_saved" if any(d["crop_status"] == "saved" for d in detections) else
                "no_valid_crops"
            )
            manifest["frames"].append(frame_result)
    finally:
        sampled.close()
    if not manifest["frames"]:
        raise ValueError("Video contains no decodable frames")
    manifest["saved_faces"] = total_faces
    manifest["status"] = "complete" if total_faces else "no_usable_faces"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return total_faces


def main():
    parser = argparse.ArgumentParser(description="Sample video frames and save YuNet face crops with metadata.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_YUNET_MODEL)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    detector = create_yunet_detector(args.model, args.threshold)
    total_faces = process_video(args.video, args.output, detector, args.interval, args.model)
    print("Total faces saved:", total_faces)
    print("Manifest:", args.output / "manifest.json")


if __name__ == "__main__":
    main()

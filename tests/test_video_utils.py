"""Regression checks for sampling, crop safety, colors, and manifest behavior."""
import json

import cv2
import numpy as np
import pytest

from src import video_utils as vu


@pytest.fixture
def blank_video(tmp_path):
    path = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 64))
    assert writer.isOpened(), "MJPG video encoder is required for this fixture"
    try:
        for _ in range(9):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_sampler_preserves_source_indices_and_timestamps(blank_video):
    samples = list(vu.iter_sampled_frames(blank_video))
    assert [index for index, _, _ in samples] == [0, 4, 8]
    assert [timestamp for _, timestamp, _ in samples] == [0, 1, 2]
    assert len(vu.extract_frames(blank_video)) == 3


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_sampling_interval(interval):
    with pytest.raises(ValueError, match="interval"):
        list(vu.iter_sampled_frames("unused.mp4", interval))


def test_crop_clips_negative_and_overflow_coordinates():
    frame = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
    assert vu.clip_face_box(frame, (-2.5, -1.5, 8, 7)) == (0, 0, 6, 6)
    np.testing.assert_array_equal(vu.crop_face(frame, (-2.5, -1.5, 8, 7)), frame[:6, :6])
    assert vu.clip_face_box(frame, (10, 8, 9, 9)) == (10, 8, 2, 2)


@pytest.mark.parametrize("box", [(20, 20, 2, 2), (0, 0, 0, 2), (0, 0, -1, 3), (float("nan"), 0, 2, 2)])
def test_invalid_crops_are_rejected(box):
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    assert vu.clip_face_box(frame, box) is None
    with pytest.raises(ValueError, match="valid image region"):
        vu.crop_face(frame, box)


def test_preprocess_and_png_save_preserve_color(tmp_path):
    crop = np.full((8, 12, 3), [20, 80, 220], dtype=np.uint8)
    processed = vu.preprocess_face(crop)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    np.testing.assert_allclose(processed[0, 0], np.array([220, 80, 20]) / 255)
    path = tmp_path / "crop.png"
    vu.save_face(processed, path)
    restored = cv2.imread(str(path))
    assert restored.shape == (224, 224, 3)
    np.testing.assert_array_equal(restored[0, 0], [20, 80, 220])


def test_image_write_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Could not save"):
        vu.save_face(np.zeros((224, 224, 3), dtype=np.float32), tmp_path / "fail.png")


class FixedDetector:
    """Deliberately return one clipped box and one entirely outside the image."""
    def setInputSize(self, size):
        self.input_size = size

    def detect(self, frame):
        return 1, np.array([
            [-4, -3, 20, 20, *([0] * 10), .95],
            [100, 100, 10, 10, *([0] * 10), .96],
        ], dtype=np.float32)

    def getScoreThreshold(self):
        return .85

    def getNMSThreshold(self):
        return .3

    def getTopK(self):
        return 5000


def test_pipeline_records_saved_and_rejected_crops(blank_video, tmp_path):
    output = tmp_path / "processed"
    assert vu.process_video(blank_video, output, FixedDetector()) == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["saved_faces"] == 3
    assert [f["frame_index"] for f in manifest["frames"]] == [0, 4, 8]
    for frame in manifest["frames"]:
        valid, invalid = frame["detections"]
        assert valid["crop_bbox_xywh"] == [0, 0, 16, 17]
        assert cv2.imread(str(output / valid["crop_path"])).shape == (224, 224, 3)
        assert invalid["crop_status"] == "rejected_invalid_box"
        assert invalid["crop_path"] is None
    assert len(list(output.glob("*.png"))) == 3
    with pytest.raises(FileExistsError):
        vu.process_video(blank_video, output, FixedDetector())
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_real_yunet_on_no_face_video(blank_video, tmp_path):
    detector = vu.create_yunet_detector()
    output = tmp_path / "no_faces"
    assert vu.process_video(blank_video, output, detector, model_path=vu.DEFAULT_YUNET_MODEL) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "no_usable_faces"
    assert manifest["model_sha256"] == vu.YUNET_SHA256
    assert len(manifest["frames"]) == 3
    assert all(f["status"] == "no_faces" and f["detections"] == [] for f in manifest["frames"])
    assert not list(output.glob("*.png"))


def test_missing_model_and_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        vu.create_yunet_detector(tmp_path / "missing.onnx")
    with pytest.raises(ValueError, match="Could not open video"):
        vu.get_video_info(tmp_path / "missing.mp4")

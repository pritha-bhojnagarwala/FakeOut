# YuNet face detector

`face_detection_yunet_2026may.onnx` is the small dynamic-input model from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
OpenCV Zoo documents this version for OpenCV 5.x variable-sized inputs.

- Size: 229,738 bytes.
- SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
- [Official download](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx).
- License: MIT; the upstream notice is retained in `LICENSE-YuNet.txt`.

The binary was checked against the size and SHA-256 recorded in OpenCV Zoo's
Git LFS pointer before use. It is included so preprocessing does not need a
runtime download. The default loader checks its SHA-256 before loading.
Large training checkpoints and datasets are not included.

Reference: Wu, Wei; Peng, Hanyang; Yu, Shiqi. "Yunet: A tiny millisecond-level
face detector." Machine Intelligence Research 20(5), 656–665 (2023).

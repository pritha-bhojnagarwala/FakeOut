# FakeOut
A deepfake detection system using machine learning and computer vision.

## Current implementation

Video metadata, one-second frame sampling, full-frame YuNet face detection,
safe face cropping, and a JSON preprocessing manifest. Deepfake classification,
tracking, four-branch models, and the final application are not implemented yet.

The source builds on the original local `video_utils.py`. Haar and MediaPipe
detection helpers remain available, but `process_video` now expects a YuNet
`FaceDetectorYN` instance and does not split frames into halves. MediaPipe is
optional and is imported only when its legacy helper is called.

## Setup and use

The current cloud workflow was verified with Python 3.12 and OpenCV 5.0.0
(distribution `opencv-python==5.0.0.93`). The Mac environment still needs its own
runtime verification; these preprocessing commands do not use PyTorch or MPS.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/video_utils.py --video /path/to/video.mp4 --output data/processed/run_01
```

Defaults: threshold `0.85`, interval `1` second. To override:

```bash
python src/video_utils.py --video /path/to/video.mp4 --output data/processed/run_02 --threshold 0.90 --interval 0.5
python -m pytest -q
```

Use a new or empty output directory for each run; existing results are never
overwritten. No videos or datasets are bundled. `data/` is ignored by Git.

## Outputs and limitations

The command prints the saved crop count and manifest path. Crops are named
`frame_000194_face_0.png` using the **source frame index**, not the sample number.
They are 224 x 224 PNGs. Preprocessing converts BGR to RGB float32 in `[0,1]`;
the save helper converts back to BGR bytes for correct PNG color. Future neural
networks will need their own model-specific normalization.

`manifest.json` records video metadata, detector settings and supplied model hash,
sampled frame indices and estimated timestamps, original float boxes, clipped
integer crop boxes, confidence scores, five landmarks, and relative crop paths.
It also records frames with no detections and rejected invalid boxes. A run with
no saved crops reports `no_usable_faces`; no real/fake prediction is produced.

Times are estimated from frame index / reported FPS, not verified presentation
timestamps, and can be inaccurate for variable-frame-rate footage. Frames outside
the sampling schedule are not analyzed. Detection order is not a person/track ID.
Face scores are uncalibrated detection scores, not manipulation probabilities.
Occlusion and faces near image boundaries can cause misses.

## Pilot evidence

Prior to integration, the bundled YuNet model at `0.85` detected all 30 visible
face instances in 15 one-second samples of a two-speaker clip, with no visually
observed false positives. In a second clip, both `0.85` and `0.90` gave five
correct detections, no detections in five no-visible-face samples, no detections
in two heavily occluded samples, and missed two partial faces at the right edge.
These are small visual checks, not a general accuracy claim or proof of
superiority to BlazeFace. The direct detector comparison was stopped before
BlazeFace inference at the user's request. No comparative result is claimed.

See [models/README.md](models/README.md) for the model source and license.

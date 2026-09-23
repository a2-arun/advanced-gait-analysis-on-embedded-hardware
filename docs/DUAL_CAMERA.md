# Dual-camera capture (branch: `dual-camera-support`)

Two USB webcams at different angles instead of one, aimed at improving
robustness on the Jetson Nano: if the subject is occluded, side-on, or
poorly framed from one angle, the other can still produce a usable gait
sequence.

## What this is - and isn't

This is **not** multi-view fusion into a single feature vector. The
checkpoint (`models/checkpoint/full_hybrid_best.pth`) was trained on
single-view 78-dim sequences; feeding it a fused two-camera representation
would run it outside the input distribution it was validated on (see
`docs/ARCHITECTURE.md` #4-#5 for why this project is already careful about
that). Building a real multi-view model would mean retraining on the
research team's side, which is out of scope here.

What's implemented instead: **both cameras run the full pipeline in
parallel** (their own `GaitFeatureExtractor` + `LiveSequenceBuffer`), and
each identification pass uses whichever camera's sequence has the higher
`pose_quality` (fraction of frames with a detected pose) once both are
ready, or whichever is ready after a short grace period if the other hasn't
completed yet. See `src/identification/dual_identifier.py`.

## Why 480p, not 720p

Two cameras compete for the same constraints as one, twice over:

- **USB bandwidth.** The Jetson Nano's USB ports are shared internally; two
  webcams streaming uncompressed video at 720p can saturate that well
  before the cameras' own frame-rate limits do. `DualCameraManager`
  requests the **MJPG** fourcc explicitly (`src/camera/dual_camera_manager.py`),
  which is roughly 3-5x smaller over the wire than raw YUYV at the same
  resolution - most USB webcams support it natively, no extra CPU cost for
  the encode (the camera's ISP does it).
- **CPU.** MediaPipe pose extraction (CPU-bound) is already this project's
  latency bottleneck on a single camera (see `docs/ARCHITECTURE.md` #7).
  Running it on two streams roughly **doubles that cost** - there's no way
  around this without skipping frames, since accuracy through robustness is
  the entire point of the second camera. Keeping each stream at 480p
  instead of 720p is what makes that doubled cost survivable on the Nano's
  quad-core CPU.

`config.yaml`'s `camera.resolution` (`[640, 480]`) applies to both cameras
via `camera.secondary_device_id`. Don't raise it for a dual-camera setup
without first measuring `processing_time_ms` (see the README's "where the
output is" section) at 480p and confirming there's headroom.

## Setup

1. Plug in both USB webcams. On the Jetson, `ls /dev/video*` to find their
   indices (usually `0` and `1`, but don't assume - unplug/replug one at a
   time if unsure which is which).
2. In `config.yaml`, set `camera.secondary_device_id` (it defaults to
   `null`, which keeps every existing single-camera script working
   unchanged).
3. `python scripts/test_dual_camera.py --no-display` - confirms both open,
   reports combined fps, and saves a frame from each to
   `outputs/dual_camera_test_{primary,secondary}.jpg`.
4. Enrollment is unchanged (`scripts/enroll.py` still uses one camera - the
   primary). Run identification with both cameras via:
   ```
   python scripts/identify_dual.py --device-id jetson-nano-01
   ```

## Status

Implemented and syntax/import-checked, and the two-stream capture and
threading path was exercised against synthetic video sources to confirm
frames from both cameras arrive independently and stay synced. **Not yet
run against two physical cameras** - this development machine only has one
webcam. Before relying on this on the Nano: run `scripts/test_dual_camera.py`
first, then a short `scripts/identify_dual.py` session, and watch
`tegrastats` for CPU/thermal headroom under the doubled pose-extraction
load.

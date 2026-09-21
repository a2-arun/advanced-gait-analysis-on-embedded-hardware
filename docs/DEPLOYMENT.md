# Jetson Nano Deployment

**Status: install path written, not yet run on the device.** The step-by-step
install is in the "Jetson Nano" section of the top-level `README.md`
(uses `requirements-jetson.txt`). This document keeps the background: what
was risky and why the choices there were made. The risk list below is the
original pre-hardware analysis; where README.md and this file disagree,
README.md is current.

## What "Jetson Nano" means here

The original **Jetson Nano** (2019, 4GB, Maxwell GPU, quad-core
Cortex-A57) tops out at **JetPack 4.6.x**, which means **Ubuntu 18.04**,
**Python 3.6** by default, **CUDA 10.2**, **cuDNN 8.2**, **TensorRT 8.2**.
NVIDIA does not support this board on JetPack 5/6 (those target the Orin
family). Every compatibility question below is scoped to that JetPack
4.6.x / Ubuntu 18.04 / Python 3.6 stack - if the board turns out to be an
Orin Nano instead, most of this changes (newer Ubuntu, newer Python, and
the compatibility problems below mostly disappear).

## Known compatibility risks (unverified until hardware is available)

These are risks to check for, not confirmed blockers - don't pre-solve them
without the actual device in hand, per the project's own ground rules.

1. **mediapipe's Python wheels and Python 3.6.** The Tasks API this project
   uses (`mediapipe.tasks.python.vision.PoseLandmarker`) needs a fairly
   recent mediapipe (this project developed against 0.10.21), and recent
   mediapipe releases require Python >= 3.8. JetPack 4.6's stock Python is
   3.6. Likely paths, in order of effort:
   - Install a newer Python (3.8+) alongside the system one via
     `pyenv`/`deadsnakes`-style builds, keeping the system Python untouched.
   - Fall back to mediapipe's older `mp.solutions.pose` legacy API, which
     had wider unofficial ARM/Jetson build support, and adjust
     `src/pose/pose_extraction.py` accordingly (it isolates all
     MediaPipe-specific calls in one class for exactly this reason).
   - Replace pose estimation entirely with NVIDIA's `trt_pose`, built
     specifically for Jetson via TensorRT. Bigger change (different
     landmark set), only worth it if mediapipe genuinely can't be made to
     work.

2. **PyTorch on aarch64 + JetPack 4.6.** Use NVIDIA's official Jetson
   PyTorch wheels (published on the NVIDIA developer forums / Jetson Zoo),
   not a generic `pip install torch` - generic PyPI wheels are not built
   for Jetson's ARM+CUDA combination. These wheels top out around PyTorch
   1.10-1.13 for JetPack 4.6. The model here is small (~850K params,
   `models/full_pipeline.py`) and uses only standard layers (Conv1d, LSTM,
   TransformerEncoder) - nothing exotic - so an older PyTorch should be
   able to load the checkpoint and run inference; this needs confirming on
   the real device, not assumed.

3. **OpenCV.** JetPack images usually ship a pre-built OpenCV with GStreamer
   support for the CSI camera connector - prefer that over `pip install
   opencv-python`, which lacks GStreamer/CSI camera support.

## What ports directly vs. what needs adaptation

| Component | Expectation |
|---|---|
| `models/*.py`, checkpoint | Ports as-is - it's just PyTorch module code |
| `src/pose/pose_extraction.py` | Logic ports; the MediaPipe backend may need swapping (see risk #1) |
| `src/features/`, `src/model/`, `src/database/`, `src/events/` | Pure Python + numpy/sqlite3 - ports as-is |
| `src/camera/camera_manager.py` | May need a GStreamer capture pipeline string for a CSI camera instead of a plain device index |
| `.venv` / `requirements.txt` | Does NOT port - Jetson needs its own pinned set per the risks above |

## Do not, until there's a device to test against

- Upgrade JetPack, CUDA, cuDNN, or TensorRT versions
- Replace NVIDIA-provided drivers or libraries
- Install a generic (non-Jetson) PyTorch wheel
- Assume any of the risks above are solved

## When the Nano arrives: first commands to run

```bash
cat /etc/nv_tegra_release        # JetPack/L4T version
python3 --version
nvcc --version                    # CUDA version
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
tegrastats                        # live CPU/GPU/RAM/temp - use this for the
                                   # performance comparison table this project
                                   # still owes (see docs/ARCHITECTURE.md #7)
```

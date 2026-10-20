# Nano setup and capture notes

Follow the one-time [README setup](../README.md), then use its daily two-script workflow. The supported target is original
Nano / JetPack 4.6.x / Python 3.6, with NVIDIA PyTorch 1.10 and the explicitly
experimental MediaPipe 0.8.5 community wheel. The exact device installation remains
to be validated. Modern `requirements.txt` is for workstation artifact preparation
and tests; `requirements-jetson.txt` is for the Nano.

## Camera choices

The `capture` command records a separate USB/V4L2 take at 640x480/30 requested FPS.
The same capture helper is used automatically by `enroll.py` and `identify.py`
when `--video` is omitted. Camera timestamps are retained in the encoded file rather than synthesized from
pose detections. Actual timestamps and continuity are rechecked during inference.
Configure USB device/resolution/FPS in `config.yaml`. The two scripts use those
defaults; `--device /dev/video0` overrides the device explicitly. Run
`python scripts/check_nano.py --capture-test` to check package versions, CUDA,
media tools and a private temporary 2-second USB encode/decode, then
`python scripts/gait.py check --require-nano` for the real pose/model smoke test.
These checks must pass on the actual Nano before declaring it validated.

For a CSI camera, first test a JetPack GStreamer pipeline. The following illustrative
pipeline records locally using sensor timestamps; confirm the sensor mode, plugin
availability, framing and file PTS on your actual Nano:

```bash
umask 077
mkdir -p data/raw
test ! -e data/raw/csi_take1.mp4 && \
  gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  nvv4l2h264enc ! h264parse ! qtmux ! filesink location=data/raw/csi_take1.mp4
```

Stop after the intended independent take with Ctrl+C so `-e` flushes the muxer.
`gst-launch-1.0` / `nvv4l2h264enc` come from the JetPack GStreamer installation;
install the matching JetPack components if absent. This CSI command has not been
executed on hardware here. Register/enroll/verify the resulting file using the same
commands as a USB or phone original. Never overwrite it to fix orientation or timing.
For camera preview only, `camera.device_id` can instead hold an `nvarguscamerasrc`
GStreamer pipeline ending in BGR `appsink`; verification still reads the saved file.

## Common failures

- CUDA assertion fails: use the NVIDIA wheel matching JetPack/Python/aarch64 and
  inspect CUDA libraries. The runtime fails instead of sending video elsewhere.
- No `mp.solutions.pose`: the installed package is not the required MediaPipe 0.8.5
  recipe. A newer Tasks API is a different integration and must not be substituted.
- Missing lite model: use the tagged asset instructions in README, then rerun the
  pose graph check. Changing assets changes the recipe and requires recalibration.
- EGL/display error from the community wheel: validate its GPU/GL environment locally;
  a monitor/X session may be needed by that build. SSH alone does not prove a pose
  graph can start. Do not assert headless compatibility without running the check.
- `INSUFFICIENT_GAIT`: inspect full body/feet, lighting, motion, timing and continuity.
  Record another independent take; do not fill missed poses or call the result FALSE.
- `CLAIM_NOT_ENROLLED`: enroll that ID using the current recipe. Recipe changes require
  a new store and re-enrollment from the preserved originals.
- `POLICY_NOT_READY`: this is expected before Nano-specific calibration/evidence.
- Decoder error: check `ffprobe -show_streams -show_frames VIDEO`. The port refuses
  oversized or ambiguous media; do not silently replace the original with an edit.

Keep consent/name mappings outside the public repository. Local captures/templates,
model bundles and outputs are git-ignored. Encrypt the data volume and apply an
appropriate retention policy if required by the study; file permissions alone are
not encryption. No telemetry or remote inference is implemented.

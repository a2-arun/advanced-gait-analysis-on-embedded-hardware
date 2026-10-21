# Next lab visit: simple checklist

## The big idea (read this first)

We do **not** have the final walking-recognition brain yet. So this visit is **not** about
recognising anyone. It is about checking that the *body* of the Nano is ready: the computer
itself, the camera, the screen and the place where records will be saved. When the final brain
arrives later, we just plug it in.

Think of it like setting up a new TV before the movie arrives. Today we only check that the
screen, the speakers and the cables work.

**Do not:** enrol any real person, try to get a TRUE/FALSE answer, or add fake people to the
database. The tool will refuse anyway, and that is correct.

## Bring with you

- The Nano, power supply, USB camera, HDMI screen, keyboard.
- A USB stick with `dist/jetson-lab-kit-20261021.tar.gz` and its `.sha256` file.
- A phone or notebook to write down what you see.
- Only people who have agreed to be filmed, if you film anyone at all. A plain object or empty
  room is fine for the camera tests.

## Steps (in order)

Open a terminal on the Nano. Run each command, then write down what it said, even if it looks fine.

### 1. Unpack the kit
```bash
sha256sum -c jetson-lab-kit-*.tar.gz.sha256 && tar xzf jetson-lab-kit-*.tar.gz
cd jetson-lab-kit-*
python3 scripts/build_lab_kit.py verify .
```
*Why:* makes sure the files were not damaged while copying. You want to see `"ok": true`.

### 2. Ask the Nano who it is
```bash
cat /proc/device-tree/model; echo
cat /etc/nv_tegra_release
python3 --version
free -m
df -h .
```
*Why:* we have been *assuming* it is a 4 GB original Nano with the right software. Now we
find out for real. Write down every line.

### 3. Run the health check
```bash
python3 scripts/lab.py preflight
```
*Why:* one command that checks everything it can and saves a report. Each line says PASS,
FAIL, WARNING, NOT_TESTED or BLOCKED.
- **PASS** = good. **FAIL** = something is wrong, read the reason next to it.
- **BLOCKED on "model_artifacts"** = expected! It means "no final brain installed yet".
- **NOT_TESTED** on the Nano = the check could not run, write down which one.

If lots of things FAIL because software is missing (PyTorch, MediaPipe, etc.), **stop and note
it**. Do not try to fix it by upgrading things at random.

### 4. Test the camera
```bash
python3 scripts/lab.py camera list
python3 scripts/lab.py camera record --seconds 5
python3 scripts/lab.py camera analyze --video data/diagnostics/camera/<the file it just made>.mp4
```
*Why:* proves the camera works, shows what picture sizes it supports, and checks the video
is smooth and not black or frozen. These clips are only test clips; they are kept separate
from anything that will ever be used to recognise a person.

### 5. Test the body-point finder (pose)
```bash
python3 scripts/lab.py pose check
python3 scripts/lab.py pose extract --video data/diagnostics/camera/<same clip>.mp4 \
    --overlay outputs/lab-reports/overlay-1.mp4
```
*Why:* the Nano will find body points (shoulders, knees, feet) in the video. We check that this
part starts, then **look at the overlay video yourself** to see whether the dots sit on the
body. The computer's "confidence" number does not prove the dots are in the right place, so
your eyes matter here. Ideally film someone walking (with their OK).

### 6. Create the empty database
```bash
python3 scripts/lab.py db check
python3 scripts/lab.py db init
python3 scripts/lab.py db check
```
*Why:* makes the empty place where records will be saved later. It should say "0 references".
Nobody is added.

### 7. Test the screen
```bash
python3 scripts/lab.py display status --render window --watch 5
```
*Why:* shows a status page on the HDMI screen. Expected message: **NOT READY: MODEL NOT
DEPLOYED**. Check that you can read it from where you will stand. If it complains there is no
desktop, start the desktop and try again, or use `--render png --output outputs/lab-reports/status.png`.

### 8. Look at the empty "brain slot"
```bash
python3 scripts/lab.py model status
```
*Why:* confirms the Nano knows it has no approved final model and will refuse to guess.
It should say `MODEL_NOT_DEPLOYED`.

### 9. Save the evidence and tidy up
```bash
tar czf ~/nano-evidence-$(date +%Y%m%d).tar.gz outputs/lab-reports data/diagnostics
```
Copy that file to the USB stick. Delete test clips you do not need
(`rm -i data/diagnostics/camera/*.mp4`), then shut down properly:
```bash
sync && sudo shutdown -h now
```
*Why:* the saved reports are the proof of what really worked on the real Nano. We cannot
learn this from the laptop.

## What to write down / bring back

- The outputs from steps 2 and 3 (most important).
- Anything that said FAIL or WARNING, with the exact words.
- Whether the camera picture looked normal, and how many frames per second it reported.
- Whether the dots on the overlay video sat on the body.
- Whether the screen text was readable.
- The evidence file from step 9.

## If something goes wrong

- Camera not found: unplug and replug it, try another USB port, run step 4 again.
- "Permission" problem on the camera: ask before changing user groups; note it down.
- Pose check fails: usually the wrong MediaPipe version. Note the version it reports.
- Anything confusing: stop, keep the output, do not change system software.
- Full troubleshooting table: `docs/JETSON_NANO_LAB_RUNBOOK.md`.

## What happens after

1. You bring back the evidence. We fix whatever the real Nano showed us (missing software,
   camera settings, screen layout).
2. The research side decides the final walking-recognition model.
3. That model is delivered as a package and installed with the steps in
   `docs/FINAL_MODEL_ACTIVATION_RUNBOOK.md`. Only then do we enrol people and test real
   TRUE/FALSE answers, and only after a proper decision rule has been worked out from evidence.

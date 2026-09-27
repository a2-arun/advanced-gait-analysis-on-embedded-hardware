# Gait Edge Identification

Identifies people by the way they walk, from a live webcam, using the
pretrained GaitGraph2 gait model. This page covers testing it on a laptop.
Jetson Nano setup is in [docs/JETSON.md](docs/JETSON.md); how the model
works and how accurate it is is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 1. One-time setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Get the model weights (not in git):

1. Download `model_weights.zip` from
   https://github.com/tteepe/GaitGraph2/releases/tag/v0.1
2. Convert it (a zip or an extracted folder both work):

```bash
python scripts/convert_gaitgraph2.py path\to\model_weights.zip
```

Check everything works (no camera needed):

```bash
python tests/test_gait_model.py        # expect: [OK] all GaitModel checks passed
python scripts/test_camera.py          # live camera preview, press q to close
```

## 2. Set up the space

- Put the laptop at about waist height with a clear walking path **straight
  toward it**, starting 4-5 m back.
- Everyone walks **toward the camera**, both when enrolling and when
  identifying. Side-on walks are much less accurate, and enrolling one way
  then identifying the other doesn't work.
- Keep your whole body, feet included, in view as long as possible. Use
  decent light and plain clothes (no long coats or bags).

## 3. Enroll people

Start from an empty database (this deletes everyone enrolled so far):

```bash
del database\gait.db               # macOS/Linux: rm database/gait.db
```

Enroll each person:

```bash
python scripts/enroll.py --name "Abhishek"
```

A preview window shows your skeleton: a green border means you're tracked,
red means you're not.

1. Walk toward the laptop at your normal pace. A pass takes about 2-3 s
   (the `Frames` counter fills up), then "Pass N captured" flashes.
2. Step out of view and go back to the start **out of view**. A walk away
   from the camera would be captured as a pass.
3. Repeat until **5 passes** are captured. Enrollment then saves by itself.
   Pressing `q` cancels without saving.

"Pass rejected" or "Pass too short" means redo that pass: tracking dropped
out, or you reached the camera too soon (start further back).

Repeat for everyone, e.g. `--name "Arhaan"`. Enroll at least 2 people.

## 4. Identify

```bash
python scripts/identify.py
```

Walk toward the camera the same way as when enrolling. After each 2-3 s
walk the terminal prints one of:

```
IDENTIFIED: Abhishek  (similarity: 0.9123)
  other candidates: Arhaan=0.4410

UNKNOWN PERSON  (best candidate similarity: 0.6012, threshold: 0.7500)
```

Stop with `q` in the window or Ctrl+C. Ignore results from walking back to
the start.

Test all three cases: each enrolled person, plus someone who isn't enrolled.

## 5. Tune the threshold

`similarity_threshold` in `config.yaml` (default 0.75) decides between a name
and UNKNOWN. It was measured on different videos, so your webcam probably
needs its own value. Note the scores from a few walks each:

- **Strangers get a name** -> raise it above their scores.
- **Enrolled people come up UNKNOWN** -> lower it below their scores.

Pick a value between the two groups. No re-enrollment needed; just restart
`identify.py`.

## What to expect

On the 13-person research videos (walking toward the camera, 2 people
enrolled), it picks the right one of the two ~94% of the time, and about 1
in 4 strangers still gets a name at the default threshold. It's a demo, not
access control. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

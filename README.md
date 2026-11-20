# MudraAI

Recognising classical dance hand mudras from a webcam or an image. I use MediaPipe to get the hand keypoints, turn them into a few geometric features, and train an XGBoost model on top. Trained on the Sattriya mudra dataset (29 classes).

This is a baseline. It looks at one frame at a time, so there's no motion or temporal info yet.

## How it works

1. MediaPipe finds 21 landmarks on the hand.
2. The landmarks are shifted so the wrist is the origin and scaled by hand size, so position and distance from the camera don't matter.
3. From those I compute 33 features: 13 joint angles and 20 distances (fingertip to wrist, tip to tip, and a few others).
4. XGBoost predicts the mudra.
5. A One-Class SVM sits in front of it and rejects poses that don't look like any mudra, otherwise the classifier would always guess something.

## Results

About 81% accuracy on the held-out test split. Some mudras get mixed up a lot (Singhamukha, Krishnasarmukha, Dhanu, Sarpashira).

The dataset images are tiny (200x200), and MediaPipe misses the hand on a lot of them. I upscale and pad the image and retry, which gets detection up to around 82%. The rest are skipped.

## Setup

Python 3.11.

```
pip install -r requirements.txt
```

Download the Sattriya dataset and put the folder next to this one. The path is set in `config.py`.

## Running it

```
python 01_prepare_data.py       # images -> csv files
python 02_train_classifier.py   # trains XGBoost
python 03_train_gatekeeper.py   # trains the one-class SVM
python 04_webcam_test.py        # live demo
```

In the demo, press `g` to toggle the gatekeeper and `q` to quit.

## Files

- `config.py` - paths and settings
- `features.py` - landmark normalisation and the 33 features
- `01_prepare_data.py` - extracts features, splits train/test, adds noisy copies for training
- `02_train_classifier.py` - classifier
- `03_train_gatekeeper.py` - anomaly filter
- `04_webcam_test.py` - webcam demo

There's a longer walkthrough in `DOCUMENTATION.md`.

## Notes

- `mediapipe==0.10.21` is pinned on purpose. Newer versions (1.x) removed the `mp.solutions.hands` API that this code uses.
- Test set is small (a couple hundred samples), so treat the accuracy number as rough.

## Ideas for later

- Use video and model the motion between frames
- Try a neural network on the landmarks
- Collect more data with real webcam images

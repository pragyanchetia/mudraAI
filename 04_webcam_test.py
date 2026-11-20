"""Step 4: live webcam demo. Keys: g = toggle gatekeeper, q/ESC = quit."""
import sys

import cv2
import joblib
import mediapipe as mp
import numpy as np

import config as C
from features import extract_features, landmarks_to_array, normalize_landmarks


def load_models():
    if not C.CLASSIFIER_PATH.exists() or not C.LABEL_ENCODER_PATH.exists():
        print("Model files not found. Run 02_train_classifier.py first.")
        sys.exit(1)
    classifier = joblib.load(C.CLASSIFIER_PATH)
    encoder = joblib.load(C.LABEL_ENCODER_PATH)
    gatekeeper = joblib.load(C.GATEKEEPER_PATH) if C.GATEKEEPER_PATH.exists() else None
    print(f"Loaded classifier ({len(encoder.classes_)} classes); "
          f"gatekeeper: {'yes' if gatekeeper else 'no'}")
    return classifier, encoder, gatekeeper


def main():
    classifier, encoder, gatekeeper = load_models()
    use_gate = gatekeeper is not None

    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access the webcam.")
        sys.exit(1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text, color = "No hand detected", (128, 128, 128)

        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            norm = normalize_landmarks(landmarks_to_array(hl))
            if norm is not None:
                x = np.array([extract_features(norm)])
                if use_gate and gatekeeper.predict(x)[0] != 1:
                    text, color = "Unknown / non-mudra pose (rejected)", (0, 0, 255)
                else:
                    probs = classifier.predict_proba(x)[0]
                    idx = int(np.argmax(probs))
                    conf = probs[idx] * 100
                    text = f"{encoder.inverse_transform([idx])[0]} ({conf:.1f}%)"
                    color = (0, 200, 0) if conf >= 60 else (0, 165, 255)

        cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 85), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 85), color, 2)
        cv2.putText(frame, text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        gate = "ON" if use_gate else "OFF"
        cv2.putText(frame, f"Gatekeeper: {gate} (g toggle, q quit)", (25, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("MudraAI - Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("g") and gatekeeper is not None:
            use_gate = not use_gate

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()

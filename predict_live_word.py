import torch
import torch.nn as nn
import numpy as np
import cv2
import imutils
import dlib
from imutils import face_utils
from pynput import keyboard

from model_one_word import Net

# ------------------------------------
# SETTINGS
# ------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "word_model_gpu.pth"
SHAPE_PREDICTOR_PATH = "../sentence_lip_reading/shape_predictor_68_face_landmarks.dat"


LABELS = ['bin', 'lay', 'place', 'set']
MAX_FRAMES = 29

triggered = False


# ------------------------------------
# Keyboard trigger
# ------------------------------------
def on_press(key):
    global triggered
    try:
        if key.char == 's':
            triggered = True
            print("[INFO] Capturing word...")
    except:
        pass


# ------------------------------------
# MAIN
# ------------------------------------
if __name__ == "__main__":

    print("Using device:", DEVICE)

    # Load face detector
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    # Load model
    print("[INFO] Loading trained model...")
    model = Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    cap = cv2.VideoCapture(0)
    print("[INFO] Webcam started. Press S to predict. Q to quit.")

    frame_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=900)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            mouth = shape[48:68]
            (x, y, w, h) = cv2.boundingRect(np.array([mouth]))

            mouth_roi = frame[y-50:y+h+50, x-50:x+w+50]

            cv2.rectangle(frame,
                          (x-50, y-50),
                          (x+w+50, y+h+50),
                          (0, 255, 0), 2)

            if triggered:
                frame_buffer.append(mouth_roi)

                if len(frame_buffer) == MAX_FRAMES:
                    triggered = False

                    # Preprocess
                    video = []

                    for f in frame_buffer:
                        f = cv2.resize(f, (128, 64))
                        f = f.astype(np.float32) / 255.0
                        video.append(f)

                    video = np.array(video)  # (T,H,W,C)
                    video = torch.from_numpy(video)
                    video = video.permute(3, 0, 1, 2)  # (C,T,H,W)
                    video = video.unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(video)
                        pred_idx = torch.argmax(output, dim=1).item()

                    print("Prediction:", LABELS[pred_idx])

                    frame_buffer = []

        cv2.imshow("Word Lip Reading", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

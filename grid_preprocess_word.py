import os
import cv2
import numpy as np
import dlib
from imutils import face_utils

# ---------------------------------------
# SETTINGS
# ---------------------------------------

GRID_PATH = r"D:/GRID CORPUS"
SAVE_PATH = r"D:/grid_word_dataset"


WORDS = ['bin', 'lay', 'place', 'set']
MAX_FRAMES = 29

SHAPE_PREDICTOR_PATH = "../sentence_lip_reading/shape_predictor_68_face_landmarks.dat"

# ---------------------------------------

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

os.makedirs(SAVE_PATH, exist_ok=True)

total_saved = 0

print("Starting preprocessing...\n")

# ---------------------------------------
# LOOP THROUGH SPEAKERS
# ---------------------------------------

for speaker in os.listdir(GRID_PATH):

    speaker_path = os.path.join(GRID_PATH, speaker)

    if not os.path.isdir(speaker_path):
        continue

    print("Processing speaker:", speaker)

    for file in os.listdir(speaker_path):

        if not file.endswith(".mpg"):
            continue

        video_path = os.path.join(speaker_path, file)
        align_path = video_path.replace(".mpg", ".align")

        if not os.path.exists(align_path):
            continue

        # ---------------------------------------
        # READ ALIGN FILE
        # ---------------------------------------

        with open(align_path, "r") as f:
            lines = f.readlines()

        first_word = None

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            word = parts[2].lower()

            if word != "sil":
                first_word = word
                break

        if first_word not in WORDS:
            continue

        # ---------------------------------------
        # PROCESS VIDEO
        # ---------------------------------------

        cap = cv2.VideoCapture(video_path)

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                mouth = shape[48:68]
                (x, y, w, h) = cv2.boundingRect(np.array([mouth]))

                mouth_roi = frame[y-50:y+h+50, x-50:x+w+50]

                mouth_roi = cv2.resize(mouth_roi, (128, 64))

                frames.append(mouth_roi)

        cap.release()

        if len(frames) == 0:
            continue

        # ---------------------------------------
        # FIX LENGTH TO 29 FRAMES
        # ---------------------------------------

        if len(frames) > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]
        elif len(frames) < MAX_FRAMES:
            pad = MAX_FRAMES - len(frames)
            last = frames[-1]
            frames.extend([last] * pad)

        video_np = np.array(frames)

        # ---------------------------------------
        # SAVE
        # ---------------------------------------

        word_folder = os.path.join(SAVE_PATH, first_word)
        os.makedirs(word_folder, exist_ok=True)

        save_name = f"{speaker}_{file.replace('.mpg', '.npy')}"
        save_path = os.path.join(word_folder, save_name)

        np.save(save_path, video_np)

        total_saved += 1

print("\nDONE.")
print("Total samples saved:", total_saved)

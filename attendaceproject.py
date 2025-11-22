"""
Attendance system (Version 2) using face_recognition (dlib ResNet 128D embeddings).
"""

import os
import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ENCODINGS_PATH = os.path.join(MODELS_DIR, "encodings.pkl")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "Attendance.csv")

# Parameters
TOLERANCE = 0.45
FRAME_RESIZE_SCALE = 0.25
RECOGNITION_COOLDOWN = 5

def ensure_dirs():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        print(f"Created '{IMAGES_DIR}'. Add training images and rerun.")
        return False
    return True

def create_attendance_file():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w") as f:
            f.write("Name,Time,Date\n")

def mark_attendance(name):
    with open(ATTENDANCE_CSV, "r+") as f:
        lines = f.readlines()
        names_present = [line.split(",")[0] for line in lines]
        if name not in names_present:
            now = datetime.now()
            f.write(f"{name},{now.strftime('%H:%M:%S')},{now.strftime('%Y-%m-%d')}\n")
            print(f"[Attendance] Marked: {name}")

def encode_faces_from_images(force_reencode=False):
    if not ensure_dirs():
        raise SystemExit("Add images and rerun.")

    if os.path.exists(ENCODINGS_PATH) and not force_reencode:
        print("Loading existing encodings...")
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(image_files) == 0:
        print("No images found in images/")
        raise SystemExit

    known_encodings = []
    known_names = []

    print(f"[Training] Found {len(image_files)} images...")

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        person_name = os.path.splitext(img_name)[0]
        base_person = person_name.split("_")[0]

        print(f"[Training] Reading {img_name}...")

        # Load using cv2 for safety
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[Error] Cannot read file {img_name}, skipping...")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if rgb.dtype != np.uint8:
            print(f"[Error] Invalid image type in {img_name}, skipping...")
            continue

        face_locations = face_recognition.face_locations(rgb, model="hog")
        if len(face_locations) == 0:
            print(f"[Warning] No face found in {img_name}, skipping...")
            continue

        face_encs = face_recognition.face_encodings(rgb, face_locations)
        for enc in face_encs:
            known_encodings.append(enc)
            known_names.append(base_person)

        print(f"[Training] Processed {img_name} → {base_person}")

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"[Training] Saved {len(known_encodings)} encodings.")
    return known_encodings, known_names

def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        return None, None
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_and_mark(known_encodings, known_names):
    last_seen = {}
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        return

    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        # FIX 1: If frame is empty, skip
        if not ret or frame is None:
            print("Warning: Empty frame, skipping...")
            continue

        # FIX 2: Must be uint8 image
        if frame.dtype != np.uint8:
            print("Warning: Invalid frame type, skipping...")
            continue

        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)

        # FIX 3: Validate small frame
        if small_frame is None or small_frame.dtype != np.uint8:
            print("Warning: Invalid resized frame, skipping...")
            continue

        # Convert to RGB safely
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Ensure contiguous array and uint8 type for dlib
        rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)

        # Face detection
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_enc)
            name = "UNKNOWN"

            if len(distances) > 0:
                best_idx = np.argmin(distances)
                if distances[best_idx] < TOLERANCE:
                    name = known_names[best_idx]

            # Scale back to original frame size
            top = int(top / FRAME_RESIZE_SCALE)
            right = int(right / FRAME_RESIZE_SCALE)
            bottom = int(bottom / FRAME_RESIZE_SCALE)
            left = int(left / FRAME_RESIZE_SCALE)

            color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if name != "UNKNOWN":
                now = time.time()
                if now - last_seen.get(name, 0) > RECOGNITION_COOLDOWN:
                    mark_attendance(name.upper())
                    last_seen[name] = now

        cv2.imshow("Attendance System (V2)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_attendance_file()

    force_reencode = False
    encs, names = load_encodings()

    if encs is None or names is None or force_reencode:
        encs, names = encode_faces_from_images(force_reencode)

    print(f"[Main] Loaded encodings for {len(set(names))} people.")

    recognize_and_mark(encs, names)

import face_recognition
import cv2
import os
import pandas as pd
from datetime import datetime

def load_known_faces():
    known_encodings, known_names = [], []
    for person in os.listdir("data/faces"):
        for img_name in os.listdir(f"data/faces/{person}"):
            img_path = f"data/faces/{person}/{img_name}"
            img = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(person)
    return known_encodings, known_names

def recognize_and_log(known_encodings, known_names):
    cap = cv2.VideoCapture(0)
    logged = []

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        for enc, loc in zip(encodings, faces):
            matches = face_recognition.compare_faces(known_encodings, enc)
            name = "Unknown"

            if True in matches:
                name = known_names[matches.index(True)]
                if name not in logged:
                    log_attendance(name)
                    logged.append(name)

            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def log_attendance(name):
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = now.strftime("%Y-%m-%d")

    if not os.path.exists("data/attendance.csv"):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.read_csv("data/attendance.csv")

    df = pd.concat([df, pd.DataFrame([[name, date, time]], columns=df.columns)])
    df.to_csv("data/attendance.csv", index=False)

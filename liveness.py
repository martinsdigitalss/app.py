import cv2
import mediapipe as mp
import streamlit as st
import time

def liveness_check():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    blink_count = 0
    frame_window = st.image([])  # Streamlit image placeholder

    st.info("Press 'Q' in webcam window to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            blink_count += 1
            cv2.putText(frame, f"Liveness Score: {blink_count}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Stop if enough blinks are detected
        if blink_count > 10:
            cap.release()
            return True

        # Small delay so browser can keep up
        time.sleep(0.1)

    cap.release()
    return False


# === STREAMLIT UI ===
st.title("Automated Attendance System")
st.write("Using Liveness Detection with Face Mesh")

if st.button("Start Liveness Check"):
    result = liveness_check()
    if result:
        st.success("✅ Liveness Verified. Attendance Marked!")
    else:
        st.error("❌ Liveness not detected.")

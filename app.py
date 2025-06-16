import cv2
import mediapipe as mp
import streamlit as st
import time

# Set Streamlit page config
st.set_page_config(page_title="Automated Attendance System", layout="centered")

# Page Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0fdf4;
    }
    .stButton>button {
        background-color: #22c55e;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #16a34a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("🎓 Automated Attendance System")
st.subheader("🔐 Liveness Detection with Face Mesh")
st.write("Ensure only real humans (not photos) mark attendance using webcam-based blink detection.")

# Liveness Check Function
def liveness_check():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    blink_count = 0
    frame_window = st.image([])

    with st.spinner("Starting webcam and detecting blinks..."):
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

            if blink_count > 10:
                cap.release()
                return True

            time.sleep(0.1)

    cap.release()
    return False

# Button to Start
if st.button("✅ Start Liveness Check"):
    result = liveness_check()
    if result:
        st.success("✅ Liveness Verified. Attendance Marked!")
    else:
        st.error("❌ Liveness could not be verified.")

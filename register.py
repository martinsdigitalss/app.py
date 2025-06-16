import cv2
import os

name = input("Enter student name: ")
cam = cv2.VideoCapture(0)
count = 0
os.makedirs(f"data/faces/{name}", exist_ok=True)

while count < 10:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Register - Press S to save", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"data/faces/{name}/{count}.jpg", frame)
        count += 1

cam.release()
cv2.destroyAllWindows()

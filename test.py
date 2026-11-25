import cv2
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    if ok:
        cv2.putText(frame, "Camera " + str(i), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Camera " + str(i), frame)
    cap.release()
cv2.waitKey(0)

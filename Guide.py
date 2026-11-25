import cv2, time, threading, pyttsx3

speaking = False

def speak(text):
    global speaking
    if speaking:
        return
    def run():
        global speaking
        speaking = True
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        speaking = False
    threading.Thread(target=run, daemon=True).start()

detector = cv2.QRCodeDetector()
cap = cv2.VideoCapture(1)   # change to the camera number that showed your iPhone
last_msg, last_time = "", 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    text, points, _ = detector.detectAndDecode(frame)

    if points is None:
        msg = "Can't see the code, sweep slowly"
    else:
        pts = points[0]
        cx, cy = pts.mean(axis=0)
        size = cv2.contourArea(pts.astype("float32")) / (w * h)
        cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 3)

        if cx < w * 0.4:    msg = "Move left"
        elif cx > w * 0.6:  msg = "Move right"
        elif cy < h * 0.4:  msg = "Tilt up"
        elif cy > h * 0.6:  msg = "Tilt down"
        elif size < 0.08:   msg = "Move closer"
        elif size > 0.40:   msg = "Move back"
        else:               msg = "Hold still"

    # speak at most once every 1.2 seconds, never repeat the same message
    if msg != last_msg and time.time() - last_time > 1.2:
        print(msg)
        speak(msg)
        last_msg, last_time = msg, time.time()

    cv2.putText(frame, last_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Guidance", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
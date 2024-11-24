import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Kamera görüntüsü alınamadı!")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id in [0, 4, 8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)

    # FPS hesaplama
    cTime = time.time()
    time_diff = cTime - pTime
    fps = 1 / time_diff if time_diff > 0 else 0
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 75), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)

    # Görüntü gösterimi
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çıkış
        break

cap.release()
cv2.destroyAllWindows()

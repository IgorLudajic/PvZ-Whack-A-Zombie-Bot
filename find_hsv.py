import cv2
import numpy as np
import mss
import ctypes

try:
    ctypes.windll.user32.SetProcessDPIAware()
except: pass

def nothing(x):
    pass

cv2.namedWindow('HSV Podesavanje')
cv2.createTrackbar('H Min', 'HSV Podesavanje', 15, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Podesavanje', 150, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Podesavanje', 150, 255, nothing)
cv2.createTrackbar('H Max', 'HSV Podesavanje', 35, 179, nothing)
cv2.createTrackbar('S Max', 'HSV Podesavanje', 255, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Podesavanje', 255, 255, nothing)

sct = mss.mss()
monitor = sct.monitors[1]

print("Pomeraj slajdere dok ne izolujes SUNCE (da bude belo, ostalo crno).")
print("Pritisni 'q' kad zavrsis da vidis vrednosti.")

while True:
    img = np.array(sct.grab(monitor))
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos('H Min', 'HSV Podesavanje')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Podesavanje')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Podesavanje')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Podesavanje')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Podesavanje')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Podesavanje')

    lower_yellow = np.array([h_min, s_min, v_min])
    upper_yellow = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    preview = cv2.resize(img_bgr, (600, 450))
    mask_preview = cv2.resize(mask, (600, 450))
    
    cv2.imshow('Original', preview)
    cv2.imshow('HSV Maska', mask_preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n=============================")
        print("KOPIRAJ OVE VREDNOSTI U GAMEPLAY.PY:")
        print(f"LOWER_YELLOW = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"UPPER_YELLOW = np.array([{h_max}, {s_max}, {v_max}])")
        print("=============================\n")
        break

cv2.destroyAllWindows()
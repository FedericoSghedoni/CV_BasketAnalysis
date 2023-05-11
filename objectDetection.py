import cv2
import pyvirtualcam
import pyautogui

# Crea la webcam virtuale
with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    # Ciclo di acquisizione e invio dei frame
    while True:
        # Acquisisce uno screenshot dello schermo
        screen = cv2.cvtColor(
            cv2.resize(
                pyautogui.screenshot(),
                (1280, 720)
            ),
            cv2.COLOR_BGR2RGB
        )

        # Invia il frame alla webcam virtuale
        cam.send(screen)

        # Aggiorna la webcam virtuale
        cam.sleep_until_next_frame()

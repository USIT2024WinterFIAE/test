import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

# Initialisierung von MediaPipe für die Handgestenerkennung
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

class HandTrackingGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Tracking mit LED-Anzeige")
        
        # Erstellen der Hauptframes
        self.video_frame = tk.Frame(window)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.led_frame = tk.Frame(window)
        self.led_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Video-Label für die Kamera-Anzeige
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()
        
        # LED-Canvas erstellen
        self.leds = []
        led_names = ["Daumen", "Zeigefinger", "Mittelfinger", "Ringfinger", "Kleiner"]
        
        for i, name in enumerate(led_names):
            frame = tk.Frame(self.led_frame)
            frame.pack(pady=5)
            
            # LED (Ein Kreis auf einem Canvas)
            canvas = tk.Canvas(frame, width=50, height=50)
            canvas.pack(side=tk.LEFT, padx=5)
            led = canvas.create_oval(5, 5, 45, 45, fill='gray')
            self.leds.append(led)
            
            # Label für den Fingernamen
            tk.Label(frame, text=name).pack(side=tk.LEFT, padx=5)
            
            # Speichern des Canvas für späteren Zugriff
            canvas.led = led
            self.leds[i] = canvas
        
        # Kamera initialisieren
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Kamera konnte nicht geöffnet werden.")
            self.window.quit()
        
        self.update()
    
    def detect_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]
        thumb_tip = 4
        finger_states = [0, 0, 0, 0, 0]

        # Thumb detection
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            finger_states[0] = 1

        # Finger detection
        for idx, tip in enumerate(finger_tips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                finger_states[idx + 1] = 1

        return finger_states
    
    def update_leds(self, finger_states):
        for i, state in enumerate(finger_states):
            # LED-Farbe aktualisieren (grün für an, grau für aus)
            color = 'green' if state == 1 else 'gray'
            self.leds[i].itemconfig(self.leds[i].led, fill=color)
    
    def update(self):
        success, image = self.cap.read()
        if success:
            # Bild spiegeln und für MediaPipe vorbereiten
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # Hand Landmarks zeichnen und Finger erkennen
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    finger_states = self.detect_fingers(hand_landmarks)
                    self.update_leds(finger_states)
            
            # Bild für GUI vorbereiten und anzeigen
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((640, 480))
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        
        # Nächstes Frame nach 10ms
        self.window.after(10, self.update)
    
    def release_camera(self):
        if hasattr(self, 'cap'):
            self.cap.release()
    
    def __del__(self):
        self.release_camera()

# Hauptprogramm
if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.release_camera)
    root.mainloop()
import cv2
import face_recognition
import numpy as np
import sqlite3
import json
import requests
import time
import datetime
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
from dotenv import load_dotenv
from picamera2 import Picamera2
from libcamera import Transform

load_dotenv()

# Backend API endpoints
API_PREFIX = os.getenv("API_PREFIX", "https://q9z46niut5.execute-api.ap-southeast-1.amazonaws.com")
API_FETCH_EMPLOYEES = API_PREFIX + "/employee"
API_UPLOAD_ACTIVITY = API_PREFIX + "/activity/sync"
MACHINE_ID = int(os.getenv("MACHINE_ID", 1))

# Global flag to control recognition state
recognition_active = True

# Load employee data from backend (unchanged)
def sync_employee_data():
    print("[SYNC] Starting to sync employee data from backend...")
    response = requests.get(API_FETCH_EMPLOYEES)
    if response.status_code == 200:
        employees = response.json()

        conn = sqlite3.connect("employees.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM employees")

        for emp in employees:
            cursor.execute("INSERT INTO employees (id, name, face_encoding) VALUES (?, ?, ?)",
                           (emp["id"], f"{emp['firstName']} {emp['lastName']}", emp["faceId"]))
        conn.commit()
        conn.close()
        print("[SYNC] Employee data synced successfully.")
    else:
        print(f"[SYNC] Failed to fetch employees. Status code: {response.status_code}")

# Rest of the helper functions remain the same (load_employee_data, get_status, record_attendance, upload_and_sync)

# GUI Application
class FaceRecognitionApp:
    def __init__(self, root):
        print("[INIT] Initializing FaceRecognitionApp...")
        self.root = root
        self.root.title("Employee Face Recognition System")
        self.root.attributes("-fullscreen", True)

        # Set vertical resolution
        screen_width = 320
        screen_height = 480

        # Initialize PiCamera2 with correct color space and resolution
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={
                "size": (screen_height, screen_width),  # Swapped for vertical orientation
                "format": "RGB888"  # Explicitly set color format to RGB
            },
            transform=Transform(hflip=1, vflip=1)  # Optional: adjust if camera is inverted
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()

        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height)
        self.canvas.pack()

        self.status_label = tk.Label(self.canvas, text="", font=("Helvetica", 24), bg="black", fg="white")
        self.status_label.place(x=20, y=20)

        self.last_detected = time.time()
        self.update_frame()

    def update_frame(self):
        if recognition_active:
            # Capture frame from PiCamera2
            frame = self.picam2.capture_array()
            
            # Ensure correct color space (this step is crucial)
            # Use cv2.cvtColor to convert from RGB to BGR if needed
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            
            # Rotate the frame 90 degrees for vertical orientation
            rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
            
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized = False
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)
                if True in matches:
                    match_index = matches.index(True)
                    emp_id = self.known_face_ids[match_index]
                    emp_name = self.known_face_names[match_index]

                    if time.time() - self.last_detected > 3:
                        self.last_detected = time.time()
                        record_attendance(emp_id, emp_name)
                        self.status_label.config(text=f"Recognized: {emp_name}")
                        self.last_recognized_time = time.time()
                    recognized = True

            # Clear label after 2 seconds
            if self.status_label.cget("text") and time.time() - self.last_recognized_time > 2:
                self.status_label.config(text="")

            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.root.after(16, self.update_frame)

    def reload_face_data(self):
        print("[RELOAD] Reloading face data after sync...")
        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()
        print("[RELOAD] Reload completed.")

# The rest of the script (setup_local_db and main block) remains unchanged

# Remaining code is the same as in the previous version

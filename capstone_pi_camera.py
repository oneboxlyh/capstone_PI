import face_recognition
import numpy as np
import sqlite3
import json
import requests
import time
import datetime
import os
import threading
import queue
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
from dotenv import load_dotenv
from picamera2 import Picamera2, Preview
from libcamera import Transform

load_dotenv()

# Backend API endpoints
API_PREFIX = os.getenv("API_PREFIX", "https://q9z46niut5.execute-api.ap-southeast-1.amazonaws.com")
API_FETCH_EMPLOYEES = API_PREFIX + "/employee"
API_UPLOAD_ACTIVITY = API_PREFIX + "/activity/sync"
MACHINE_ID = int(os.getenv("MACHINE_ID", 1))

# Global flag to control recognition state with lock for thread safety
recognition_active = True
recognition_lock = threading.Lock()

# Debug flag - set to True to see more debug information
DEBUG_MODE = True

# Face detection parameters - HIGHLY TUNABLE FOR DIFFERENT ENVIRONMENTS
DETECTION_ZOOM = 1.0      # Increase to zoom in (1.5 means 50% zoom)
DOWNSAMPLE_FACTOR = 2     # Higher means faster but less accurate detection 
FACE_UPSCALE = 1          # Higher means better detection of small faces but slower
MATCH_TOLERANCE = 0.6     # Higher value = more permissive matching (0.6 is permissive)

# Image enhancement parameters
BRIGHTNESS_FACTOR = 1.2   # Increase brightness (1.0 = normal)
CONTRAST_FACTOR = 1.3     # Increase contrast (1.0 = normal)
SHARPNESS_FACTOR = 1.5    # Increase sharpness (1.0 = normal)

# Load employee data from backend
def sync_employee_data():
    print("[SYNC] Starting to sync employee data from backend...")
    try:
        response = requests.get(API_FETCH_EMPLOYEES, timeout=10)
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
            print(f"[SYNC] Employee data synced successfully. Total employees: {len(employees)}")
        else:
            print(f"[SYNC] Failed to fetch employees. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[SYNC] Network error during sync: {e}")
    except Exception as e:
        print(f"[SYNC] Error during sync: {e}")

# Load employee face encodings from local DB
def load_employee_data():
    print("[LOAD] Loading employee face data from local DB...")
    try:
        conn = sqlite3.connect("employees.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, face_encoding FROM employees")
        employees = cursor.fetchall()
        conn.close()

        known_face_encodings = []
        known_face_ids = []
        known_face_names = []

        for emp in employees:
            known_face_ids.append(emp[0])
            known_face_names.append(emp[1])
            known_face_encodings.append(np.array(json.loads(emp[2])))

        print(f"[LOAD] Employee face data loaded. Total employees: {len(employees)}")
        return known_face_ids, known_face_names, known_face_encodings
    except Exception as e:
        print(f"[LOAD] Error loading employee data: {e}")
        return [], [], []

# Determine status (CHECK IN or CHECK OUT)
def get_status(emp_id):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect("employees.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE employee_id = ? AND DATE(timestamp) = ?", (emp_id, today))
        count = cursor.fetchone()[0]
        conn.close()
        return "CHECK IN" if count % 2 == 0 else "CHECK OUT"
    except Exception as e:
        print(f"[STATUS] Error getting status: {e}")
        return "CHECK IN"  # Default to CHECK IN if there's an error

# Record attendance in a separate thread
def record_attendance_thread(emp_id, emp_name, status_queue):
    try:
        print(f"[RECORD] Recording attendance for {emp_name}...")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = get_status(emp_id)

        conn = sqlite3.connect("employees.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (employee_id, name, status, timestamp) VALUES (?, ?, ?, ?)",
                       (emp_id, emp_name, status, timestamp))
        conn.commit()
        conn.close()
        
        print(f"[RECORD] Attendance recorded for {emp_name} as {status}.")
        # Put the result in the queue for the main thread to process
        status_queue.put((emp_name, status))
    except Exception as e:
        print(f"[RECORD] Error recording attendance: {e}")
        status_queue.put(None)  # Signal an error occurred

# Upload and clean attendance
def upload_and_sync():
    global recognition_active
    print("[UPLOAD] Starting attendance upload and sync...")
    
    # Safely disable recognition during sync
    with recognition_lock:
        recognition_active = False

    try:
        conn = sqlite3.connect("employees.db")
        cursor = conn.cursor()
        cursor.execute("SELECT employee_id, name, status, timestamp FROM attendance")
        logs = cursor.fetchall()

        if logs:
            batch_size = 30
            for i in range(0, len(logs), batch_size):
                batch = logs[i:i + batch_size]
                activities = []
                for log in batch:
                    activities.append({
                        "employee_id": log[0],
                        "status": log[2],
                        "created_at": log[3],
                        "machine_id": MACHINE_ID
                    })
                try:
                    response = requests.post(API_UPLOAD_ACTIVITY, json={"activities": activities}, timeout=10)
                    if response.status_code == 201:
                        print(f"[UPLOAD] Batch {i//batch_size + 1} uploaded successfully.")
                    else:
                        print(f"[UPLOAD] Batch {i//batch_size + 1} failed. Status code: {response.status_code}")
                except Exception as e:
                    print(f"[UPLOAD] Exception during upload: {e}")

            cursor.execute("CREATE TABLE IF NOT EXISTS attendance_bk AS SELECT * FROM attendance WHERE 0")
            cursor.execute("INSERT INTO attendance_bk SELECT * FROM attendance")
            cursor.execute("DELETE FROM attendance")
            conn.commit()

        conn.close()
    except Exception as e:
        print(f"[UPLOAD] Error during upload: {e}")

    # Load new face data in a separate thread to avoid blocking
    threading.Thread(target=sync_and_reload_faces, args=(app,), daemon=True).start()
    
    # Re-enable recognition
    with recognition_lock:
        recognition_active = True
    
    print("[UPLOAD] Attendance upload and sync completed.")
    # Schedule the next sync
    threading.Timer(600, upload_and_sync).start()

def sync_and_reload_faces(app_instance):
    try:
        sync_employee_data()
        app_instance.reload_face_data()
    except Exception as e:
        print(f"[SYNC] Error during sync and reload: {e}")

# Enhance image for better face detection
def enhance_image(image):
    # Convert to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
        
    # Apply brightness enhancement
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(BRIGHTNESS_FACTOR)
    
    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(CONTRAST_FACTOR)
    
    # Apply sharpness enhancement
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(SHARPNESS_FACTOR)
    
    # Convert back to numpy array if needed
    if isinstance(image, np.ndarray):
        return np.array(pil_image)
    
    return pil_image

# Process frame in a separate thread
def process_frame_thread(frame, app_instance):
    try:
        # Get RGB channels only
        rgb_frame = frame[:, :, :3]
        
        # Make a copy we can modify
        detection_frame = rgb_frame.copy()
        
        # Enhance the image for better detection
        detection_frame = enhance_image(detection_frame)
        
        # Apply zoom if enabled
        height, width = detection_frame.shape[:2]
        if DETECTION_ZOOM > 1.0:
            # Calculate crop dimensions
            new_width = int(width / DETECTION_ZOOM)
            new_height = int(height / DETECTION_ZOOM)
            
            # Calculate crop coordinates to center the face
            start_x = (width - new_width) // 2
            start_y = (height - new_height) // 2
            
            # Crop the image to zoom in
            detection_frame = detection_frame[start_y:start_y+new_height, start_x:start_x+new_width]
        
        # Downsample for speed - adjust factor as needed
        if DOWNSAMPLE_FACTOR > 1:
            small_frame = detection_frame[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
        else:
            small_frame = detection_frame
        
        # Only proceed if recognition is active
        with recognition_lock:
            if not recognition_active:
                app_instance.processing = False
                return

        # Run face detection
        face_locations = face_recognition.face_locations(
            small_frame, 
            model="hog",
            number_of_times_to_upsample=FACE_UPSCALE
        )
        
        # If no faces found, try again with higher upscale (slower but more sensitive)
        if not face_locations and FACE_UPSCALE < 2:
            if DEBUG_MODE:
                print("[DEBUG] First detection pass failed, trying with higher upscale")
            face_locations = face_recognition.face_locations(
                small_frame, 
                model="hog",
                number_of_times_to_upsample=2
            )
        
        if DEBUG_MODE:
            print(f"[DEBUG] Found {len(face_locations)} faces")
        
        # Scale face locations back to original frame size
        scale_factor = DOWNSAMPLE_FACTOR
        
        # Calculate offsets if zoom was applied
        x_offset = 0
        y_offset = 0
        if DETECTION_ZOOM > 1.0:
            x_offset = (width - int(width / DETECTION_ZOOM)) // 2
            y_offset = (height - int(height / DETECTION_ZOOM)) // 2
        
        # Store face locations for visualization, scaling back to original image
        app_instance.face_locations = [
            (top * scale_factor + y_offset, 
             right * scale_factor + x_offset, 
             bottom * scale_factor + y_offset, 
             left * scale_factor + x_offset) 
            for top, right, bottom, left in face_locations
        ]
        
        # Update the debug info on screen
        current_face_count = len(face_locations)
        app_instance.face_count = current_face_count
        
        # Update UI message based on whether faces are detected
        current_time = time.time()
        
        # THIS IS THE FIX: Only show "No face detected" if face_count is actually 0
        if not face_locations:
            if current_time - app_instance.last_recognized_time > 3:
                app_instance.root.after(0, lambda: app_instance.status_label.config(
                    text="No face detected",
                    fg="yellow"
                ))
            app_instance.processing = False
            return
        else:
            # If faces are found, update message to reflect this
            if current_time - app_instance.last_recognized_time > 3:
                app_instance.root.after(0, lambda: app_instance.status_label.config(
                    text="Face detected, checking...",
                    fg="cyan"
                ))
        
        # Process face encodings since faces were found
        face_encodings = face_recognition.face_encodings(
            small_frame, 
            face_locations
        )
        
        # Flag to track if we found a match
        match_found = False
        
        for face_encoding in face_encodings:
            # Try with our configurable tolerance
            matches = face_recognition.compare_faces(
                app_instance.known_face_encodings, 
                face_encoding, 
                tolerance=MATCH_TOLERANCE
            )
            
            if True in matches:
                match_index = matches.index(True)
                emp_id = app_instance.known_face_ids[match_index]
                emp_name = app_instance.known_face_names[match_index]

                # Check the recognition cooldown
                if current_time - app_instance.last_detected > 5:
                    app_instance.last_detected = current_time
                    # Put a task on the queue to record attendance in yet another thread
                    threading.Thread(
                        target=record_attendance_thread, 
                        args=(emp_id, emp_name, app_instance.status_queue),
                        daemon=True
                    ).start()
                    
                match_found = True
                # We found a match, no need to check other faces
                break
                
        # If face was detected but no match found in database, show invalid face message
        if not match_found and current_time - app_instance.last_recognized_time > 3:
            app_instance.root.after(0, lambda: app_instance.status_label.config(
                text="Invalid face detected",
                fg="orange"
            ))
    except Exception as e:
        print(f"[PROCESS] Error processing frame: {e}")
    finally:
        # Mark processing as complete
        app_instance.processing = False

# GUI Application using PiCamera
class FaceRecognitionApp:
    def __init__(self, root):
        print("[INIT] Initializing FaceRecognitionApp with PiCamera...")
        self.root = root
        self.root.title("Employee Face Recognition System")
        self.root.attributes("-fullscreen", True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Initialize PiCamera2
        self.camera = Picamera2()
        
        # Configure camera with better settings for face detection
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480)},  # Good balance of resolution and performance
            transform=Transform(hflip=True)  # Mirror image for more natural interaction
        )
        self.camera.configure(config)
        
        # Additional camera settings if possible
        try:
            # Set auto white balance mode to auto (0)
            self.camera.set_controls({"AwbMode": 0})
            # Set auto exposure mode to auto (0) 
            self.camera.set_controls({"AeMode": 0})
            # Set brightness to a slightly higher value for better face detection
            self.camera.set_controls({"Brightness": 0.1})
        except Exception as e:
            print(f"[CAMERA] Warning: Could not set some camera parameters: {e}")
        
        self.camera.start()
        
        # Allow camera to warm up and adjust exposure
        time.sleep(2)

        # Threading control variables
        self.processing = False
        self.status_queue = queue.Queue()
        
        # Face detection variables
        self.face_count = 0
        self.face_locations = []
        self.show_boxes = True
        
        # Load employee data
        self.known_face_ids, self.known_face_names, self.known_face_encodings = load_employee_data()

        # Set up the UI
        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="black")
        self.canvas.pack()

        # Status message at the top
        self.status_label = tk.Label(self.canvas, text="Waiting for face...", font=("Helvetica", 24), bg="black", fg="white")
        self.status_label.place(x=20, y=20)
        
        # Debug info label
        self.debug_label = tk.Label(self.canvas, text="Face count: 0", font=("Helvetica", 16), bg="black", fg="white")
        self.debug_label.place(x=20, y=70)
        
        # Exit button in the bottom right corner
        self.exit_button = tk.Button(self.canvas, text="EXIT", font=("Helvetica", 16), 
                                     bg="red", fg="white", command=self.exit_program)
        self.exit_button.place(x=screen_width-100, y=screen_height-60, width=80, height=40)

        # Timing control
        self.last_detected = time.time()
        self.last_recognized_time = time.time()
        self.last_frame_time = time.time()
        self.last_debug_update = time.time()
        self.last_processed_time = 0
        
        # Start the frame update loop
        self.update_frame()
        
        # Start checking the status queue
        self.check_status_queue()

    def update_frame(self):
        try:
            current_time = time.time()
            
            # Rate limit frame capture to about 15 FPS
            if current_time - self.last_frame_time < 0.067:  # ~15 FPS
                self.root.after(10, self.update_frame)
                return
                
            self.last_frame_time = current_time
                
            # Only proceed with recognition logic if the flag is True
            with recognition_lock:
                is_recognition_active = recognition_active
                
            if is_recognition_active:
                # Capture frame from PiCamera
                frame = self.camera.capture_array()
                
                # Convert to PIL image for display
                pil_image = Image.fromarray(frame[:, :, :3])
                
                # Enhance the image for display
                enhanced_image = enhance_image(pil_image)
                
                # Draw face boxes if enabled
                if self.show_boxes and self.face_locations:
                    draw = ImageDraw.Draw(enhanced_image)
                    for top, right, bottom, left in self.face_locations:
                        # Draw the rectangle
                        draw.rectangle([(left, top), (right, bottom)], outline="lime", width=3)
                
                # Resize for display
                display_image = enhanced_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
                
                # Display the frame
                imgtk = ImageTk.PhotoImage(image=display_image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk
                
                # Update debug info (every half second at most)
                if current_time - self.last_debug_update > 0.5:
                    self.debug_label.config(text=f"Face count: {self.face_count}")
                    self.last_debug_update = current_time
                
                # Process frame in a separate thread if not already processing
                # Limit processing to a reasonable rate (5 FPS for detection)
                if not self.processing and (current_time - self.last_processed_time) >= 0.2:
                    self.processing = True
                    self.last_processed_time = current_time
                    threading.Thread(
                        target=process_frame_thread, 
                        args=(frame.copy(), self),
                        daemon=True
                    ).start()
            
            # Schedule the next frame update
            self.root.after(10, self.update_frame)
            
        except Exception as e:
            print(f"[UPDATE] Error updating frame: {e}")
            # Still try to continue
            self.root.after(100, self.update_frame)

    def check_status_queue(self):
        try:
            # Check if there's anything in the queue
            while not self.status_queue.empty():
                result = self.status_queue.get_nowait()
                if result:
                    emp_name, status = result
                    # Update UI from the main thread with success message
                    self.status_label.config(
                        text=f"{emp_name} successfully {status}",
                        fg="lime"  # Green color for success
                    )
                    self.last_recognized_time = time.time()
            
            # Schedule the next check
            self.root.after(100, self.check_status_queue)
        except Exception as e:
            print(f"[QUEUE] Error checking status queue: {e}")
            self.root.after(100, self.check_status_queue)

    def reload_face_data(self):
        print("[RELOAD] Reloading face data after sync...")
        try:
            new_ids, new_names, new_encodings = load_employee_data()
            
            # Update the face data in a thread-safe way
            with recognition_lock:
                self.known_face_ids = new_ids
                self.known_face_names = new_names
                self.known_face_encodings = new_encodings
                
            print("[RELOAD] Reload completed.")
        except Exception as e:
            print(f"[RELOAD] Error reloading face data: {e}")
        
    def exit_program(self):
        print("[EXIT] Exiting program on user request...")
        self.cleanup()
        self.root.destroy()
        
    def cleanup(self):
        print("[CLEANUP] Stopping camera...")
        try:
            self.camera.stop()
            print("[CLEANUP] Camera stopped.")
        except Exception as e:
            print(f"[CLEANUP] Error during cleanup: {e}")

# Initialize local database
def setup_local_db():
    print("[DB] Setting up local database if not exists...")
    try:
        conn = sqlite3.connect("employees.db")
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            face_encoding TEXT NOT NULL)""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL)""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS attendance_bk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL)""")

        conn.commit()
        conn.close()
        print("[DB] Database setup completed.")
    except Exception as e:
        print(f"[DB] Error setting up database: {e}")

if __name__ == "__main__":
    print("[MAIN] Application starting...")
    
    # Set up the database
    setup_local_db()
    
    # Do initial sync
    sync_employee_data()

    # Create and run the application
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    
    # Handle proper cleanup when window is closed
    def on_closing():
        app.cleanup()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the sync timer
    threading.Timer(600, upload_and_sync).start()
    
    print("[MAIN] Face recognition started.")
    root.mainloop()

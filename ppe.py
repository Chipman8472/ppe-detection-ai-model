import cv2
import threading
import pyttsx3
import os
import tkinter as tk
from tkinter import Label, Button, Canvas, Listbox, Scrollbar
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime

# ===== CONFIGURABLE SETTINGS =====
MODEL_PATH = 'best.pt'   # Replace with your actual model path
SNAPSHOT_FOLDER = 'snapshots'                  # Folder where violation images will be saved
CONFIDENCE_THRESHOLD = 0.5                      # Minimum confidence to accept detection

# Create snapshot folder if it doesn't exist
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Load the trained YOLOv8 model
model = YOLO(MODEL_PATH)

# Set up the TTS engine
engine = pyttsx3.init()

# Global variables for video capture and GUI components
cap = None
running = False  # Flag to control video feed
canvas = None
alert_label = None
log_listbox = None

# ======= ALERT & LOGGING FUNCTIONS =======
def alert_worker():
    engine.say("Warning! PPE Violation Detected!")
    engine.runAndWait()

def log_violation(class_name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {class_name}"
    
    log_listbox.insert(tk.END, log_entry)
    log_listbox.yview(tk.END)  # Auto-scroll to bottom

    # Save to a log file as well
    with open("violation_log.csv", "a") as log_file:
        log_file.write(f"{timestamp},{class_name}\n")

def save_snapshot(frame, class_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_FOLDER}/{class_name}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

# ======= START DETECTION =======
def start_detection():
    global cap, running, canvas

    if running:
        return  # Already running

    running = True
    cap = cv2.VideoCapture("const.mp4")

    def process_video():
        global running, cap, canvas, alert_label

        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame)

            ppe_missing = False
            violation_classes = set()

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIDENCE_THRESHOLD:
                        continue  # Skip low-confidence detections

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    if class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                        ppe_missing = True
                        violation_classes.add(class_name)

                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # For other objects (optional — e.g., "person", "machinery")
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Alert and log logic
            if ppe_missing:
                alert_label.config(text="PPE ALERT: Violation Detected!", fg="red")
                threading.Thread(target=alert_worker).start()

                for cls in violation_classes:
                    log_violation(cls)
                    save_snapshot(frame, cls)
            else:
                alert_label.config(text="All clear", fg="green")

            # Convert frame to ImageTk for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas.imgtk = imgtk  # Avoid garbage collection issues

        cap.release()
        running = False

    threading.Thread(target=process_video, daemon=True).start()

# ======= STOP DETECTION =======
def stop_detection():
    global running, cap
    running = False
    if cap:
        cap.release()

# ======= GUI SETUP =======
def create_gui():
    global canvas, alert_label, log_listbox

    root = tk.Tk()
    root.title("PPE Detection System")

    # Top canvas (video feed)
    canvas = Canvas(root, width=640, height=480)
    canvas.pack()

    # Alert label
    alert_label = Label(root, text="Press Start to Begin Detection", font=("Arial", 14), fg="blue")
    alert_label.pack()

    # Buttons (Start/Stop)
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    start_button = Button(button_frame, text="Start Detection", command=start_detection, bg="green", fg="white")
    start_button.pack(side=tk.LEFT, padx=20)

    stop_button = Button(button_frame, text="Stop Detection", command=stop_detection, bg="red", fg="white")
    stop_button.pack(side=tk.RIGHT, padx=20)

    # Log panel (Listbox with Scrollbar)
    log_frame = tk.Frame(root)
    log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    scrollbar = Scrollbar(log_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    log_listbox = Listbox(log_frame, yscrollcommand=scrollbar.set, height=10, font=("Courier", 10))
    log_listbox.pack(fill=tk.BOTH, expand=True)

    scrollbar.config(command=log_listbox.yview)

    # Graceful exit
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_detection(), root.destroy()))

    root.mainloop()

# ======= RUN APP =======
if __name__ == "__main__":
    create_gui()

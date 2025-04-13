import cv2
import numpy as np
import tensorflow as tf
import csv
import os
import datetime
import time

# --- Configuration ---
# Model and Paths
MODEL_PATH = 'saved_age_gender_models/age_group_model.h5' # Path to the combined saved model
MODEL_INPUT_SIZE = (96, 96) # MUST match the training IMAGE_SIZE

# Face Detection Model (OpenCV DNN) - Download these files
# Prototxt: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
# Model: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_CONFIDENCE_THRESHOLD = 0.6 # Minimum probability to accept a face detection

# Output Logging
LOG_FILE = 'senior_citizen_log.csv'
LOG_HEADER = ['Timestamp', 'Approx_Age_Group', 'Gender']

# Age/Gender Labels (MUST match training script labels/order)
AGE_LABELS = ['0-18', '19-30', '31-45', '46-60', '61+']
SENIOR_CITIZEN_LABEL = '61+' # The label indicating > 60
GENDER_LABELS = ['Male', 'Female'] # Corresponds to model output 0 and 1

# Video Source: 0 for default webcam, or path to video file
VIDEO_SOURCE = 0 # Or "path/to/your/video.mp4"

# --- Load Models ---
try:
    print("Loading Age/Gender model...")
    age_gender_model = tf.keras.models.load_model(MODEL_PATH)
    print("Age/Gender model loaded successfully.")
except Exception as e:
    print(f"Error loading Age/Gender model from {MODEL_PATH}: {e}")
    exit()

try:
    print("Loading Face Detection model...")
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    if face_net.empty():
         raise IOError("Face detection model files not found or corrupt.")
    print("Face Detection model loaded successfully.")
except Exception as e:
    print(f"Error loading Face Detection model ({FACE_PROTO}, {FACE_MODEL}): {e}")
    print("Please ensure the .prototxt and .caffemodel files are in the same directory or provide correct paths.")
    exit()

# --- Initialize Logging ---
def initialize_log_file(filename, header):
    file_exists = os.path.isfile(filename)
    try:
        # Use 'a+' to create if not exists, and allow reading if needed (though we only append)
        with open(filename, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writerow(header)
                print(f"Created log file: {filename}")
            else:
                 print(f"Appending to existing log file: {filename}")
    except IOError as e:
        print(f"Error opening or creating log file {filename}: {e}")
        return False
    return True

if not initialize_log_file(LOG_FILE, LOG_HEADER):
    print("Exiting due to log file initialization error.")
    exit()

# --- Main Detection Loop ---
print(f"Starting video capture from source: {VIDEO_SOURCE}...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file or error reading frame.")
        break

    frame_count += 1
    (h, w) = frame.shape[:2]

    # --- Face Detection ---
    # Create blob from frame and perform forward pass
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # --- Process Detected Faces ---
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > FACE_CONFIDENCE_THRESHOLD:
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Extract face ROI
            face_roi = frame[startY:endY, startX:endX]

            # Check if ROI is valid
            if face_roi.size == 0:
                # print("Warning: Empty face ROI detected, skipping.")
                continue

            # --- Preprocess ROI for Age/Gender Model ---
            try:
                face_resized = cv2.resize(face_roi, MODEL_INPUT_SIZE)
                face_normalized = face_resized / 255.0
                face_batch = np.expand_dims(face_normalized, axis=0) # Add batch dimension

                # --- Predict Age and Gender ---
                # Prediction output is a list: [age_probs, gender_prob]
                predictions = age_gender_model.predict(face_batch)
                age_probs = predictions[0][0] # First output, first (only) batch item
                gender_prob = predictions[1][0][0] # Second output, first item, first value (sigmoid)

                # Get predicted classes
                age_index = np.argmax(age_probs)
                age_label = AGE_LABELS[age_index]

                gender_index = 1 if gender_prob > 0.5 else 0
                gender_label = GENDER_LABELS[gender_index]

                # Get current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # --- Check for Senior Citizen and Log ---
                is_senior = (age_label == SENIOR_CITIZEN_LABEL)
                log_entry = None
                if is_senior:
                    status = "Senior Citizen"
                    log_entry = [timestamp, age_label, gender_label]
                    try:
                        with open(LOG_FILE, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(log_entry)
                    except IOError as e:
                        print(f"Error writing to log file {LOG_FILE}: {e}")
                        # Decide if you want to stop or just print errors and continue
                else:
                    status = ""


                # --- Draw Bounding Box and Labels ---
                label_text = f"{gender_label}, {age_label}"
                if is_senior:
                    label_text += f" ({status})"
                    box_color = (0, 0, 255) # Red for seniors
                else:
                     box_color = (0, 255, 0) # Green for others

                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)

                # Put label text above the box
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label_text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            except Exception as e:
                print(f"Error during prediction or drawing for a face: {e}")
                # Draw a simple box indicating an error occurred for this face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 1)
                cv2.putText(frame, "Error", (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


    # --- Display FPS ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # --- Show Frame ---
    cv2.imshow("Senior Citizen Detection", frame)

    # --- Exit Condition ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: # 'q' or ESC key
        print("Exiting...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Video capture released and windows closed.")
print(f"Log file saved at: {LOG_FILE}")
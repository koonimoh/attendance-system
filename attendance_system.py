import cv2
import face_recognition
import os
import csv
import datetime
import time

def load_known_faces(known_faces_dir="known_faces"):
    known_encodings = []
    known_names = []
    # Loop through each file in the known_faces folder
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    return known_encodings, known_names

def save_mismatch_face(frame, face_location, timestamp):
    """
    Crops the mismatch face from the frame, saves it in a 'mismatches' folder,
    and returns the filename.
    """
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    # Ensure the mismatches folder exists
    mismatch_folder = "mismatches"
    os.makedirs(mismatch_folder, exist_ok=True)
    # Generate a filename using the timestamp (replace colons with hyphens)
    ts_str = timestamp.strftime("%Y-%m-%d_%I-%M-%S_%p")
    filename = os.path.join(mismatch_folder, f"mismatch_{ts_str}.jpg")
    cv2.imwrite(filename, face_image)
    return filename, (top, left, bottom, right)

def run_attendance_system():
    # Load known faces
    known_encodings, known_names = load_known_faces()

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Timing and logging variables
    start_time = time.time()         # When the program starts
    face_detected_time = None        # When a face is first detected
    log_entry = None                 # Will hold (result_name, timestamp, detected, description)
    
    # Main loop
    while True:
        now = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process detected faces if any
        if face_encodings:
            # Process only the first detected face for logging
            if log_entry is None:
                first_face_encoding = face_encodings[0]
                first_face_location = face_locations[0]
                # Compare with known faces
                matches = face_recognition.compare_faces(known_encodings, first_face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, first_face_encoding)
                best_match_index = None
                if face_distances:
                    best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])
                # Determine result based on match
                if best_match_index is not None and matches[best_match_index]:
                    result_name = known_names[best_match_index]
                    description = "Face Matched"
                else:
                    result_name = "Unknown"
                    # Save the mismatch face image
                    current_dt = datetime.datetime.now()
                    mismatch_filename, coords = save_mismatch_face(frame, first_face_location, current_dt)
                    description = f"Mismatch face captured; saved as {os.path.basename(mismatch_filename)}. Coordinates: {coords}"
                log_entry = (result_name, datetime.datetime.now(), "Yes", description)
                face_detected_time = now

            # Draw bounding boxes and labels on all detected faces
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = None
                if face_distances:
                    best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])
                display_name = "Unknown"
                if best_match_index is not None and matches[best_match_index]:
                    display_name = known_names[best_match_index]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, display_name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Attendance System", frame)

        # Allow manual exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Timing conditions:
        if log_entry is not None:
            # If a face was logged, exit 5 seconds after first detection
            if now - face_detected_time >= 5:
                print("Face detected. Exiting 5 seconds after detection.")
                break
        else:
            # If no face is detected within 10 seconds, log as no face and exit
            if now - start_time >= 10:
                print("No face detected within 10 seconds. Logging no detection and exiting.")
                log_entry = ("NF", datetime.datetime.now(), "No", "No face detected")
                break

    cap.release()
    cv2.destroyAllWindows()

    # Generate the log file name using the log_entry info
    # Use the log_entry timestamp for file naming
    ts = log_entry[1]
    date_str = ts.strftime("%Y-%m-%d")
    # Format time as "8-15-09" (using hyphens) and append AM/PM
    time_str = ts.strftime("%I-%M-%S_%p")
    # The suffix is based on the detection result: known face name, "Unknown", or "NF"
    result_suffix = log_entry[0]
    log_filename = f"{date_str}%%%%{time_str}_{result_suffix}.csv"
    print(f"Logging to file: {log_filename}")

    # Write the single log entry to the CSV file with columns: Name, Timestamp, Detected, Description
    with open(log_filename, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Name", "Timestamp", "Detected", "Description"])
        # Reformat the timestamp to a more readable format with colons for logging purposes
        timestamp_readable = ts.strftime("%I:%M:%S %p")
        csv_writer.writerow([log_entry[0], timestamp_readable, log_entry[2], log_entry[3]])
    print("Log entry written.")

if __name__ == "__main__":
    run_attendance_system()

import cv2
import face_recognition
import os
import csv
import datetime
import time  # For timing

def load_known_faces(known_faces_dir="known_faces"):
    known_encodings = []
    known_names = []
    # Loop through each file in the known_faces folder
    for filename in os.listdir(known_faces_dir):
        # Consider only image files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_faces_dir, filename)
            # Load image and encode the face (assuming one face per image)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                # Use the filename (without extension) as the person's name
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    return known_encodings, known_names

def mark_attendance(name, detected, log_file):
    """
    Log the attendance of the recognized person into a CSV file.
    The log row will contain: [Name, Timestamp, Detected]
    """
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %I-%M-%S %p")  # Using hyphens instead of colons
    with open(log_file, mode="a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([name, timestamp_str, detected])
    print(f"Logged: {name}, {timestamp_str}, {detected}")

def run_attendance_system():
    # Create a new log file for this run. The file name is based on the current time.
    log_file = datetime.datetime.now().strftime("attendance_log_%I-%M-%S_%p.csv")
    print(f"Logging to file: {log_file}")

    # Load known faces
    known_encodings, known_names = load_known_faces()

    # Open webcam (default camera)
    cap = cv2.VideoCapture(0)

    # Set up timers and logging flag:
    start_time = time.time()         # When the program starts
    face_detected_time = None        # When a face is first detected
    logged = False                   # Flag to ensure we only log once

    while True:
        now = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert frame from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find faces & encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process detected faces
        if face_encodings:
            # If we haven't logged yet, log the first recognized face
            if not logged:
                # Use the first detected face for logging
                matches = face_recognition.compare_faces(known_encodings, face_encodings[0])
                face_distances = face_recognition.face_distance(known_encodings, face_encodings[0])
                best_match_index = None
                if face_distances:
                    best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])
                name = "Unknown"
                if best_match_index is not None and matches[best_match_index]:
                    name = known_names[best_match_index]
                mark_attendance(name, "Yes", log_file)
                logged = True
                face_detected_time = now

            # Draw bounding boxes and names on each detected face
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

        cv2.imshow("Attendance System", frame)

        # Allow manual exit via 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check timing conditions:
        if logged:
            # If a face was logged, exit 5 seconds after the first detection
            if now - face_detected_time >= 5:
                print("Face detected. Exiting 5 seconds after detection.")
                break
        else:
            # If no face is detected, exit after 10 seconds total
            if now - start_time >= 10:
                print("No face detected within 10 seconds. Logging no detection and exiting.")
                mark_attendance("No Face Detected", "No", log_file)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance_system()

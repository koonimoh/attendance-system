import cv2
import face_recognition
import os
import csv
import datetime

def load_known_faces(known_faces_dir="known_faces"):
    known_encodings = []
    known_names = []

    # Loop through each file in known_faces folder
    for filename in os.listdir(known_faces_dir):
        # Consider only image files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_faces_dir, filename)
            # Load image
            image = face_recognition.load_image_file(path)
            # Encode the face (assuming one face per image)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                encoding = encodings[0]
                known_encodings.append(encoding)
                # Use the file name (without extension) as the person's name
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    return known_encodings, known_names

def mark_attendance(name, log_file="attendance_log.csv"):
    """Log the attendance of the recognized person into a CSV file."""
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Check if the person was recently logged to avoid duplicates
    # We can do this by reading the CSV and checking last log for this name.
    # For simplicity, we log every recognition event. You can refine later.
    with open(log_file, mode="a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([name, timestamp_str])

def run_attendance_system():
    # 1. Load known faces
    known_encodings, known_names = load_known_faces()

    # 2. Open webcam
    cap = cv2.VideoCapture(0)  # 0 for default laptop cam

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find faces & encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compare each found face to known faces
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Find best match
            best_match_index = None
            if len(face_distances) > 0:
                best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])

            name = "Unknown"
            if best_match_index is not None and matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)  # Log attendance here

            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance_system()

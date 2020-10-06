import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Chose an image to detect faces in
webcam = cv2.VideoCapture(0)

# Iterate over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 5)

    # Display the webcam
    cv2.resize(frame, (720, 480))
    cv2.imshow("Face Detector", frame)

    # Stop if Q is pressed
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()

print("Code Complete")

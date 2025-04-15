import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('video.mp4')

# Parameters for rectangle size
min_width_react = 80
min_height_react = 80

# Initialize the background subtractor
algo = cv2.createBackgroundSubtractorMOG2()

# Define count line position and a list to keep track of detected vehicles
count_line_position = 550  # Adjust based on video frame size
offset = 6  # Offset for counting the vehicle
detect = []
counter = 0

# Load Haar cascade for number plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect if vehicle has crossed the line
def detect_vehicles(x, y):
    global counter
    if (count_line_position - offset) < y < (count_line_position + offset):
        cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
        counter += 1
        return True
    return False

while True:
    # Read each frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    
    # Apply background subtraction
    img_sub = algo.apply(blur)
    
    # Apply dilation
    dilate = cv2.dilate(img_sub, np.ones((5, 5), np.uint8))
    
    # Define a structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Apply morphological transformation
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Detect number plates in the frame
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected number plates
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Number Plate', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Process each contour found in the frame
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        validate_contour = (w >= min_width_react) and (h >= min_height_react)
        if not validate_contour:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        detect.append((x, y))

    # Count vehicles that crossed the line
    for (x, y) in detect:
        if detect_vehicles(x, y):
            detect.remove((x, y))

    # Draw the counting line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    # Display vehicle counter
    cv2.putText(frame, f"Vehicle Counter: {counter}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Display the original video with the line and detected number plates
    cv2.imshow('Video Original', frame)
    
    # Display the processed video (foreground mask)
    cv2.imshow('Foreground Mask', dilated)
    
    # Break the loop when 'Enter' (key code 13) is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

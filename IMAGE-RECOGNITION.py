import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained smile detection model
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Specify the full path to the image file
#img_path = os.path.join(os.getcwd(), 'C:\Users\HP\Documents\PYTHON\rectangles\smiling_people.jpg')
#img_path = os.path.join(os.getcwd(), 'C:\\Users\\HP\\Documents\\PYTHON\\rectangles\\smiling_people.jpg')
img_path = os.path.join(os.getcwd(), 'smiling_people.jpg')


# Load the image to be processed
img = cv2.imread(img_path)

# Check if the image was loaded successfully
if img is None:
    print('Error: Could not read image file')
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop over each detected face
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Get the region of interest (ROI) that contains the face
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Detect smiles in the ROI
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

    # If a smile is detected, print a message
    if len(smiles) > 0:
        print('Smile detected!')

    # Loop over each detected smile
    for (sx, sy, sw, sh) in smiles:
        # Draw a rectangle around the smile
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

# Display the processed image
cv2.imshow('Smiling Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

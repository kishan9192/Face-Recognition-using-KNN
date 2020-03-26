import numpy as np
import cv2

# 0 here is a default value which stands for WebCam
# Instantiate a camera object to capture images
cam = cv2.VideoCapture(0)

# Create a haar-cascade object for face detection
# Haar cascade tells which features to extract from the face
# Inbuilt classifier of openCV
face_cas = cv2.CascadeClassifier('C:\\Users\\KISHAN MISHRA\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

# Create a placeholder for storing the data
data = []

# ix is for storing the current frame number
ix = 0

# Infinite Loop which captures an image from the camera
# cam.read() returns two values. Ret and frame
# Ret is a bool value which checks if the camera is working properly and
# is returning an object i.e. the face or not?
# Frame is a input object as a numpy matrix 
# Every image is a combination of RGB components, when we combine all the three
# matrices we get an image. Image is a collection of pixels and every pixel is a
# combination of RGB

while True:
    ret, frame = cam.read()
    if ret == True:
        # Assuming the ret variable is true, we convert the obtained frame into GrayScale,because all the functions of OpenCV on image recognition, they work on grayscale only

         # cvtColor is used convert into color. and is converting our object frame, converting the object to a GrayScale Image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        # Cascade has one more function called detectMultiScale
        # This function takes a frame, and takes a few more parameters that are used for its fine tuning. Returns an object which has all the faces it could detect in a frame
        # Every object contains the location of this image
        # So the object has a center point (x,y) and the height and width of the face that is (h,w). So 'faces' here is an object
        
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            # Extracting a particular area from the obtained image (Face_Component)
            # From first corner point y to y+h
            # From that corner point only we have x as x+w
            # and last parameter : denotes all the values of RGB in this selection
            face_component = frame[y:y+h,x:x+w, :]
            # After extracting the face component from the image
            #  Now we resize it to 50,50
            fc = cv2.resize(face_component, (50,50))
            # ix is for counting total number of captures
            # Store the face data after every 10 frames
            # only if the number of entries is < 20
            if ix%10 == 0 and len(data)<20:
                data.append(fc)
            # For visualization draw a rectangle around the face
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        ix += 1 # increment the current frame
        # Display the frame
        cv2.imshow('frame', frame)
        
        # If the user presses the Esc key (ID : 27)
        # or the number of images hits 20, we stop recording
        if cv2.waitKey(1) == 27 or len(data) >= 20:
            break
            
    else:
        # if the camera is not working, print error
        print("Error")
        
# Now we destroy all the windows we have created
cv2.destroyAllWindows()

# Convert the data to a numpy format
data = np.asarray(data)

# print the shape as a sanity check
print(data.shape)

# Save the data as a numpy matrix in an encoded format
np.save('face_03', data)

# Run the script for different people and store the data into multiple files
        

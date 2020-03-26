import numpy as np
import cv2
# Initializing the camera object and haar cascade
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('C:\\Users\\KISHAN MISHRA\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

# type of font to be used while writing the names
font = cv2.FONT_HERSHEY_SIMPLEX

# load the data from the numpy matrices and convert to linear vectors
f_01 = np.load('./face_01.npy').reshape((20, 50*50*3)) # Maya
f_02 = np.load('./face_02.npy').reshape((20, 50*50*3)) # Kishan
f_03 = np.load('./face_03.npy').reshape((20, 50*50*3)) # Dinesh

# Every image is a np matrix of shape 50*50*3
# We will convert it into a Linear Matrix of shape 50*50*3
# it will flatten the matrix, that is linearize. 
# we'll have 20 images in a linear matrix of length 50*50*3 = 7500


print(f_01.shape, f_02.shape, f_03.shape)

# outputs for one of the above statements will be (20, 7500)
# There are 20 faces each face is represented by a vector of 7500 legth
# 7500 means RGB value of every pixel
# Combining all the RGB pixels one over the other to form an image
# Now every pixel behaves as a feature and we'll use these features
# to classify the faces using our KNN classifier

names = {
    0: 'Maya',
    1: 'Kishan',
    2: 'Dinesh',
}

# Creating 60 labels
# First 20 labels = 0 for Maya
# Next 20 labels = 1 for Kishan
# Last 20 labels = 2 for Dinesh
labels = np.zeros((60, 1))
labels[:20, :] = 0.0
labels[20:40, :] = 1.0
labels[40:, :] = 2.0

# combine all info into one data array
data = np.concatenate([f_01, f_02, f_03])
print(data.shape, labels.shape) # op = (60, 7500) = training and (60,1) = labels

def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())


# x = Testing Point
# train = Training Data
# targets = Training Labels
def knn(x, train, targets, k = 5):
    m = train.shape[0]
    dist = []
    for i in range(m):
        # Compute distance from each point and store in dist[]
        dist.append(distance(x, train[i]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    # Retrieving the starting k values after getting all the labels
    # which are close to the testing point
    sorted_labels = labels[indx][:k]
    #print(sorted_labels)
    
    #print(train)
    
    # Unique function takes a list and returns the total distinct
    # values in a list with their count
    
    
    #print(np.unique(sorted_labels, return_counts = True))
    counts = np.unique(sorted_labels, return_counts = True)
    # Here we are printing the label whose count was greater in the previous step
    # count[0] has the labels that are 0, 1 here
    # count[1] contains their counts. We are printing the label which has
    # maximum value in count[1].
    return counts[0][np.argmax(counts[1])]

while True:
    # get each frame
    ret, frame = cam.read()
    
    if ret == True:
        # convert into grayscale so that we're able to apply functions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        
        # for each face
        for (x,y,w,h) in faces:
            face_component = frame[y:y+h,x:x+w, :]
            fc = cv2.resize(face_component, (50,50))
            # lab is a variable denoting the label
            # flatten function converts any given matrix into a linear vector
            # AFTER PROCESSING
            # knn function retrurns a label value, to which the testing face/point belongs to
            lab = knn(fc.flatten(), data, labels)
            # returned label is converted into a integer and passed into the dictionary to get the name
            text = names[int(lab)]
            #putText function generates the text obtained in the previous step over the box
            cv2.putText(frame, text, (x,y), font, 1, (255, 255, 0),2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == 27:
            break

    else:
        print("Error")
        
cv2.destroyAllWindows()

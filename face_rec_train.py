import cv2 as cv
import numpy as np
import os

adr = r'C:\Users\fece Detection_Recognition\Training Set'

people = sorted(os.listdir(adr))
print("People:", people)

features = []
labels = []

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for label, person in enumerate(people):
        person_path = os.path.join(adr, person)

        for image in os.listdir(person_path):
            image_path = os.path.join(person_path, image)
            img = cv.imread(image_path)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in face_rects:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv.resize(face_roi, (200, 200))
                features.append(face_roi.astype('uint8'))  
                labels.append(label)

create_train()

print(type(features))          
print(type(features[0]))       
print(features[0].dtype)      
print(features[0].shape)       
print("Features:", len(features), "Labels:", len(labels))

labels = np.array(labels, dtype='int')

# train
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

np.save('features.npy', np.array(features, dtype=object))
np.save('labels.npy', labels)

print("Training done and saved.")

print("length of of the features list = " , len(features))


print("length of labels list = " , len(labels))

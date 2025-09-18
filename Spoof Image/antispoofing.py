import os
import cv2 as cv
import numpy as np
from sklearn.svm import SVC  
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

base_train = r'C:\Users\fece Detection_Recognition\Anti_Spoofing_SET\Training'
base_val = r'C:\Users\fece Detection_Recognition\Anti_Spoofing_SET\Validation'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def extract_face(img_path):
    img = cv.imread(img_path)
    if img is None:
        return None
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in face_rects:
        face_roi = gray[y:y + h, x:x + w]
        return cv.resize(face_roi, (100, 100)).flatten()
    return None

def load_images_and_labels(base_folder):
    X, y = [], []
    label_dict = {'live': 0, 'spoof': 1}
    for label_type in ['live', 'spoof']:
        folder_path = os.path.join(base_folder, label_type)
        for person in os.listdir(folder_path):
            person_path = os.path.join(folder_path, person)
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                face = extract_face(img_path)
                if face is not None:
                    X.append(face)
                    y.append(label_dict[label_type])
    return np.array(X), np.array(y)

# anti-spoofing training
X, y = load_images_and_labels(base_train)

clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
clf.fit(X, y)

print("\nAnti-Spoofing model trained.")

# Face Recog
features = []
labels = []
people = sorted(os.listdir(os.path.join(base_train, 'live')))

def prepare_face_recognition_data():
    for label, person in enumerate(people):
        person_path = os.path.join(base_train, 'live', person)
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            face = extract_face(img_path)
            if face is not None:
                features.append(face.reshape(100, 100).astype('uint8'))
                labels.append(label)

prepare_face_recognition_data()

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, np.array(labels))

# Validation 
print("\nValidation Results")
correct = 0
total = 0
identity_total = 0
identity_correct = 0
face_total = 0
face_detected = 0

for label_type in ['live', 'spoof']:                   # Loop over both 'live' and 'spoof' folders in the validation set
    folder_path = os.path.join(base_val, label_type)
    for person in os.listdir(folder_path):             # Loop over each person's subfolder
        person_path = os.path.join(folder_path, person)
        for img_file in os.listdir(person_path):       # Loop over each image file for that person
            face_total += 1  

            img_path = os.path.join(person_path, img_file)
            face_vector = extract_face(img_path)
            if face_vector is None:
                print(f"[SKIPPED] {img_file} — No face detected.")
                continue

            face_detected += 1 
            total += 1
            
            true_label = 0 if label_type == 'live' else 1
            predicted_label = clf.predict([face_vector])[0]

            if predicted_label == true_label:
                correct += 1

            if predicted_label == 1:
                print(f"[SPOOF] {img_file}")
            else:
                if true_label == 0 and predicted_label == 0:
                    face_img = face_vector.reshape(100, 100).astype('uint8')
                    label, confidence = face_recognizer.predict(face_img)
                    predicted_person = people[label]
                    true_person = person

                    print(f"[LIVE & CORRECT]  {img_file} — Predicted: {predicted_person}, Actual: {true_person}, Confidence: {confidence:.2f}")

                    identity_total += 1
                    if predicted_person == true_person:
                        identity_correct += 1
                else:
                    print(f"[LIVE] {img_file} — But anti-spoofing misclassified.")

print("\n Anti-Spoofing Accuracy:")
print(f"Total images with detected face: {total}")
print(f"Correctly classified: {correct}")
print(f"Anti-Spoofing Accuracy: {correct / total * 100:.2f}%")

print("\n Face Recognition Accuracy:")
print(f"Total live faces predicted: {identity_total}")
print(f"Correct identities matched: {identity_correct}")
if identity_total > 0:
    print(f"Face Recognition Accuracy: {identity_correct / identity_total * 100:.2f}%")
else:
    print("No correctly classified live images were processed.")



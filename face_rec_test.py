import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

train_dir = r'C:\Users\fece Detection_Recognition\Training Set'
people = sorted(os.listdir(train_dir))  
print("People labels:", people)

val_dir = r'C:\Users\fece Detection_Recognition\Validation Set'

total_images = 0
correct_predictions = 0

for person_name in sorted(os.listdir(val_dir)):   
    person_folder = os.path.join(val_dir, person_name)

    if not os.path.isdir(person_folder):
        continue

    for img_name in sorted(os.listdir(person_folder)):   
        img_path = os.path.join(person_folder, img_name)
        image = cv.imread(img_path)

        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        total_images += 1  
        recognized = False

        for (x, y, w, h) in face_rects:
            face_roi = gray[y:y+h, x:x+w]
            
            face_roi = cv.resize(face_roi, (200, 200))  

            label, confidence = face_recognizer.predict(face_roi)
            predicted_name = people[label]

            print(f"Image: {img_name} | Actual: {person_name} | Predicted: {predicted_name} | Confidence: {confidence:.2f}")

            if predicted_name == person_name:
                correct_predictions += 1
                recognized = True
                break  # only use the first detected face

print(f"\n Correct Predictions: {correct_predictions}")
print(f" Total Images: {total_images}")
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f" Accuracy: {accuracy:.2f}%")


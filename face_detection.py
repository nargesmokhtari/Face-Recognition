
import cv2 as cv

person = cv.imread('person.jpg')

# cv.imshow("person" , person)

gray = cv.cvtColor(person , cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_rect = haar_cascade.detectMultiScale(gray , scaleFactor=1.1 , minNeighbors=3)

print("number of faces in image 1 = " , len(face_rect))

for (x,y,w,h) in face_rect:
    cv.rectangle(person , (x,y) , (x+w , y+h) , (0,255,0) , thickness=2)

# cv.imshow("Detected Faces in image 1" , person)

people = cv.imread('people.jpg')

cv.imshow("People" , people)

gray_multiple = cv.cvtColor(people , cv.COLOR_BGR2GRAY)

face_rect_multiple = haar_cascade.detectMultiScale(gray_multiple , scaleFactor=1.1 , minNeighbors=3)

print("Number of faces in image 2 = " , len(face_rect_multiple))

for (x,y,w,h) in face_rect_multiple:
    cv.rectangle(people , (x,y) , (x+w , y+h) , (0,255,0) , thickness=2)

cv.imshow("Detected Faces in image 2" , people)

cv.waitKey(0)
cv.destroyAllWindows
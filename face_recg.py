import cv2
import numpy as np
import face_recognition
import os

images = 'images'

img1 = face_recognition.load_image_file(f'{images}/policial 1_1.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file(f'{images}/policial 1_2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgFrancisco1 = face_recognition.load_image_file(f'{images}/policial 2.jpg')
imgFrancisco1 = cv2.cvtColor(imgFrancisco1, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(img1)[0]
encode1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)

faceTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (0, 0, 255), 2)

face2 = face_recognition.face_locations(imgFrancisco1)[0]
encodeFace2 = face_recognition.face_encodings(imgFrancisco1)[0]
cv2.rectangle(imgFrancisco1, (face2[3], face2[0]), (face2[1], face2[2]), (0, 0, 255), 2)

results = face_recognition.compare_faces([encode1], encodeTest)
distance = face_recognition.face_distance([encode1], encodeTest)

resultsFranciscoElon = face_recognition.compare_faces([encodeFace2], encode1)
distanceFranciscoElon = face_recognition.face_distance([encodeFace2], encode1)

print('Encode 1', encode1)
print('Encode 2', encodeFace2)

print('Teste 1 com 1', results)
print('Teste 1 com 1', distance)

print('2 com 1', resultsFranciscoElon)
print('2 com 1', distanceFranciscoElon)

cv2.imshow('1', img1)
cv2.imshow('test', imgTest)

cv2.imshow('2', imgFrancisco1)


cv2.waitKey(0)
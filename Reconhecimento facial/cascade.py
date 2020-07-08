import numpy as np
import cv2
import pickle

# usando um modelo pre-treinado para ajudar no reconhecimento
face_cascade = cv2.CascadeClassifier(
    'cascade/data/haarcascade_frontalface_alt2.xml')



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Inicializar webcam
cap = cv2.VideoCapture(0)

while True:

    # Fazer captura, a imagem obtida sera armazenada em frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecção em cima da imagem cinza
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        # (x,y) é a coordenada inferior esquesda, enquanto height e width
        # são altura e largura, respectivamente.

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:  # and conf <= 85:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)
        stroke = 2
        end_width = x + w
        end_height = y + h
        cv2.rectangle(frame, (x, y), (end_width, end_height), color, stroke)

    # Abre a imagem/frame
    cv2.imshow('frame', frame)
    # Espera até a tecla q seja pressionada
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow()

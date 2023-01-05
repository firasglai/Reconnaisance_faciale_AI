import face_recognition
import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)

# Chargez une image d'exemple et apprenez à la reconnaître.
cristiano_image = face_recognition.load_image_file("dataset/cristiano.jpg")
cristiano_face_encoding = face_recognition.face_encodings(cristiano_image)[0]

# Chargez une deuxième image d'exemple et apprenez à la reconnaître.
messi_image = face_recognition.load_image_file("dataset/messi.jpg")
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]

# Créer des tableaux d'encodages de visage connus et leurs noms
known_face_encodings = [
    cristiano_face_encoding,
    messi_face_encoding
]
known_face_names = [
    "Cristiano Ronaldo",
    "Lionel Messi"
]

# Initialiser certaines variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Prenez une seule image de vidéo
    ret, frame = video_capture.read()

    # Ne traitez que toutes les autres images de la vidéo pour gagner du temps
    if process_this_frame:
        # Redimensionnez le cadre de la vidéo à 1/4 pour un traitement plus rapide de la reconnaissance faciale
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convertir l'image de la couleur BGR (qu'OpenCV utilise) en couleur RVB (que face_recognition utilise)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Trouver tous les visages et encodages de visage dans l'image actuelle de la vidéo
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Voir si le visage correspond au(x) visage(s) connu(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

        #utiliser la face connue avec la plus petite distance à la nouvelle face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Afficher les résultats
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Mettre à l'échelle les emplacements des visages puisque le cadre dans lequel nous avons détecté a été mis à l'échelle à 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dessinez un cadre autour du visage
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Dessinez une étiquette avec un nom sous le visage
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Afficher l'image résultante
    cv2.imshow('Reconnaisance Faciale', frame)

    # Appuyez sur 'q' sur le clavier pour quitter !
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

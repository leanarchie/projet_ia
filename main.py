import cv2
from ultralytics import YOLO
import math
import numpy as np



# Charger les modèles YOLO
model_ball = YOLO("yolo11l.pt")
model_pose = YOLO("yolo11n-pose.pt")

# Charger la vidéo d'entrée
video_input_path = 'input_video2.mp4'
cap = cv2.VideoCapture(video_input_path)

# Définir les paramètres de la vidéo de sortie
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_output_path = 'output_video2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

dribble = 0

# État des mains dans la zone
stop = True

# Suivi de la position précédente du ballon
previous_y_basketball = None
ball_direction = None  # "down" ou "up"

tolerance_dist=50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prédictions YOLO
    results_ball = model_ball.predict(source=frame, conf=0.5, verbose=False)
    results_pose = model_pose.predict(source=frame, conf=0.5, verbose=False)

    basketball = None
    left_hand = None
    right_hand = None

    # Détection du ballon
    for result in results_ball:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            if model_ball.names[class_id] == 'sports ball':
                basketball = ((x1 + x2) // 2, y1)
                if previous_y_basketball is not None:
                    ball_direction = "down" if y2 > previous_y_basketball else "up"
                    print(ball_direction)
                    tolerance_dist = (x2-x1)//2
                previous_y_basketball = y2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, basketball, tolerance_dist, (0, 255, 255), 2)

    # Détection des keypoints humains
    for result in results_pose:
        keypoints = result.keypoints.xy.cpu().numpy()
        for person_keypoints in keypoints:
            if len(person_keypoints) >= 11:
                left_hand = person_keypoints[9]
                right_hand = person_keypoints[10]
                cv2.circle(frame, tuple(left_hand.astype(int)), 5, (255, 0, 0), -1)
                cv2.circle(frame, tuple(right_hand.astype(int)), 5, (0, 0, 255), -1)

    # Comparaison si toutes les données sont disponibles
    if basketball is not None and left_hand is not None and right_hand is not None and ball_direction == "down":
        x_basketball, y_basketball = basketball
        x_left_hand, y_left_hand = left_hand
        x_right_hand, y_right_hand = right_hand

        distance_left_hand_ball = math.sqrt((x_basketball - x_left_hand) ** 2 + (y_basketball - y_left_hand) ** 2)
        distance_right_hand_ball = math.sqrt((x_basketball - x_right_hand) ** 2 + (y_basketball - y_right_hand) ** 2)

        # Détection de dribble main gauche
        if (distance_left_hand_ball <= tolerance_dist or distance_right_hand_ball <= tolerance_dist) :
            print("entrée de zone")
            if not stop :
                print("dribble")
                dribble += 1
                stop = True

        # Détection de dribble main droite
        if distance_right_hand_ball > tolerance_dist and distance_right_hand_ball > tolerance_dist :
            if stop :
                print("sortie de zone")
                stop = False


    cv2.putText(frame, f"Dribbles: {dribble}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Affichage et enregistrement
    cv2.imshow("Basketball Detection and Pose Estimation", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Vidéo traitée et sauvegardée sous :", video_output_path)

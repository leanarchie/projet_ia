import cv2
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO('yolo11n.pt')  # ou votre modèle personnalisé

# Charger la vidéo d'entrée
video_input_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_input_path)

# Définir les paramètres de la vidéo de sortie
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Effectuer la détection
    results = model(frame, verbose=False)
    
    # Dessiner les boîtes de détection sur la frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            
            # Si le modèle reconnaît le ballon de basket
            if model.names[class_id] == 'sports ball':
                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame,(x1,y1+int((0.5*y1))), (x1+1,y1+int((0.5*y1))),(0, 0, 255), 2)
                y_basketball=y1

    # Écrire la frame dans la vidéo de sortie
    out.write(frame)

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Vidéo traitée et sauvegardée sous :", video_output_path)

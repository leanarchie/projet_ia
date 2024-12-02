import cv2
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")

# Initialiser la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: impossible d'accéder à la webcam")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur de lecture de la frame")
        break

    # Appliquer le modèle YOLO sur la frame capturée
    results = model(frame)

    # Accéder aux résultats des boîtes de détection
    boxes = results[0].boxes
    names = results[0].names  # Noms des classes

    # Dessiner les boxes et labels sur la frame
    for box in boxes:
        # Récupérer les coordonnées des boxes
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Dessiner un rectangle sur la frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Convertir la confiance (box.conf) en valeur flottante
        confidence = box.conf.item()  # Conversion de Tensor à valeur flottante
        
        # Ajouter le label au-dessus de la box avec la confiance
        label = f"{names[int(box.cls)]}: {confidence:.2f}"  # Le label avec la confiance
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image avec les détections en direct
    cv2.imshow("Detection Webcam", frame)

    # Quitter la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()

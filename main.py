import ultralytics
from ultralytics import YOLO

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO("yolov8n.pt")

# Détection d'objets sur une image exemple
results = model("image/vase.png")

# Inspecter les résultats et afficher les objets détectés
detected_objects = []

# Accéder aux résultats de la première image
result = results[0]  # Résultats pour la première image

# Parcourir les résultats et ajouter les objets détectés à la liste
for box, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
    class_name = result.names[int(cls)]  # Obtenir le nom de la classe
    detected_objects.append(class_name)  # Ajouter le nom de l'objet à la liste

# Afficher les objets détectés
print("Objets détectés :", detected_objects)
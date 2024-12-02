from ultralytics import YOLO
import yaml

# Chemins vers vos données et fichiers
data_yaml_path = 'data.yaml'  # Chemin vers le fichier data.yaml
train_dir = 'dataset/train'  # Dossier d'entraînement
val_dir = 'dataset/val'      # Dossier de validation

# Créer le fichier data.yaml (au besoin)
data_dict = {
    'train': train_dir,
    'val': val_dir,
    'nc': 10,  # Nombre de classes (0 à 9)
    'names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Noms des classes
}

# Sauvegarder le fichier data.yaml si il n'existe pas
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_dict, f)

# Configurer l'entraînement YOLOv5 en Python
def train_yolov5():
    
    model = YOLO("yolo11n.pt")

    # Lancer l'entraînement en mode Python
    model.train(
        data=data_yaml_path,  # Chemin vers le fichier data.yaml
        epochs=50,             # Nombre d'époques
        imgsz=640,             # Taille des images
    )

# Appeler la fonction d'entraînement
train_yolov5()

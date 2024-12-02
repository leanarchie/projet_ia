import os
import cv2

# Chemin vers votre dataset d'images
dataset_dir = "./dataset"

# Liste des classes (0 à 9)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Parcourir chaque dossier de classe
for class_id, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_dir, class_name)
    
    # Si le dossier existe, traitons les images à l'intérieur
    if os.path.isdir(class_folder):
        for img_name in os.listdir(class_folder):
            if img_name.endswith(".jpg"):
                # Chemin vers l'image
                img_path = os.path.join(class_folder, img_name)
                
                # Lire l'image pour obtenir les dimensions
                img = cv2.imread(img_path)
                height, width, _ = img.shape

                # Générer le fichier d'annotation
                txt_filename = os.path.splitext(img_name)[0] + ".txt"
                txt_path = os.path.join(class_folder, txt_filename)

                # Créer le fichier d'annotation avec une seule boîte pour chaque image (normalisation)
                with open(txt_path, "w") as f:
                    # Note: vous devrez ajuster les coordonnées si vous avez plusieurs objets dans une image
                    x_center, y_center, w, h = 0.5, 0.5, 1.0, 1.0  # Par défaut, on considère l'objet comme occupant toute l'image
                    # Normalisation des coordonnées
                    x_center /= width

import os
import shutil
import random

# Chemin vers votre dataset
dataset_dir = 'dataset'
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Créer les répertoires d'entraînement et de validation
for split_dir in [train_dir, val_dir]:
    for class_id in range(10):  # Pour les classes 0 à 9
        os.makedirs(os.path.join(split_dir, str(class_id)), exist_ok=True)

# Séparer les données
for class_id in range(10):  # Pour chaque classe
    class_folder = os.path.join(dataset_dir, str(class_id))
    images = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]

    # Mélanger aléatoirement les images
    random.shuffle(images)

    # Diviser les images en train (80%) et val (20%)
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copier les images et leurs annotations dans les répertoires train et val
    for img_name in train_images:
        shutil.copy(os.path.join(class_folder, img_name), os.path.join(train_dir, str(class_id), img_name))
        shutil.copy(os.path.join(class_folder, img_name.replace('.jpg', '.txt')), os.path.join(train_dir, str(class_id), img_name.replace('.jpg', '.txt')))
        
    for img_name in val_images:
        shutil.copy(os.path.join(class_folder, img_name), os.path.join(val_dir, str(class_id), img_name))
        shutil.copy(os.path.join(class_folder, img_name.replace('.jpg', '.txt')), os.path.join(val_dir, str(class_id), img_name.replace('.jpg', '.txt')))

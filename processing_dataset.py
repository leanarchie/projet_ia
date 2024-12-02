import os
import shutil
import random

# Dossiers de départ
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Dossiers de destination
images_train_dir = os.path.join(base_dir, 'images', 'train')
images_val_dir = os.path.join(base_dir, 'images', 'val')
labels_train_dir = os.path.join(base_dir, 'labels', 'train')
labels_val_dir = os.path.join(base_dir, 'labels', 'val')

# Créer les répertoires de destination
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)

# Fonction pour réorganiser les images et labels
def reorganize_data(source_dir, target_image_dir, target_label_dir, split_ratio=0.8):
    # Obtenir les sous-dossiers (chiffres)
    digits = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for digit in digits:
        digit_train_image_dir = os.path.join(target_image_dir, digit)
        digit_val_image_dir = os.path.join(target_image_dir.replace('train', 'val'), digit)
        digit_train_label_dir = os.path.join(target_label_dir, digit)
        digit_val_label_dir = os.path.join(target_label_dir.replace('train', 'val'), digit)
        
        # Créer les sous-dossiers pour chaque chiffre
        os.makedirs(digit_train_image_dir, exist_ok=True)
        os.makedirs(digit_val_image_dir, exist_ok=True)
        os.makedirs(digit_train_label_dir, exist_ok=True)
        os.makedirs(digit_val_label_dir, exist_ok=True)
        
        # Lister les fichiers .jpg et .txt pour chaque chiffre
        image_files = [f for f in os.listdir(os.path.join(source_dir, digit)) if f.endswith('.jpg')]
        label_files = [f.replace('.jpg', '.txt') for f in image_files]  # Assumons que les labels ont le même nom

        # Mélanger les fichiers de manière aléatoire
        data = list(zip(image_files, label_files))
        random.shuffle(data)
        
        # Diviser les données en training et validation
        split_index = int(len(data) * split_ratio)
        train_data = data[:split_index]
        val_data = data[split_index:]
        
        # Déplacer les fichiers d'entraînement
        for img, lbl in train_data:
            shutil.move(os.path.join(source_dir, digit, img), os.path.join(digit_train_image_dir, img))
            shutil.move(os.path.join(source_dir, digit, lbl), os.path.join(digit_train_label_dir, lbl))
        
        # Déplacer les fichiers de validation
        for img, lbl in val_data:
            shutil.move(os.path.join(source_dir, digit, img), os.path.join(digit_val_image_dir, img))
            shutil.move(os.path.join(source_dir, digit, lbl), os.path.join(digit_val_label_dir, lbl))

# Réorganiser les données de train et validation
reorganize_data(train_dir, images_train_dir, labels_train_dir, split_ratio=0.8)
reorganize_data(val_dir, images_val_dir, labels_val_dir, split_ratio=0.8)

# Si vous souhaitez également ajouter un dossier de test, vous pouvez ajuster la logique pour créer et déplacer vers 'test'.

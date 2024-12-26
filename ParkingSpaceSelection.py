import cv2
import json
import numpy as np

# Fonction pour charger les positions sauvegardées depuis un fichier JSON
def load_positions(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []  # Si le fichier n'est pas trouvé, retourner une liste vide

# Fonction pour sauvegarder les positions dans un fichier JSON
def save_positions(file_path, positions):
    with open(file_path, 'w') as f:
        json.dump(positions, f, indent=4)

# Liste pour stocker les points du polygone et les types de place de parking
polygon_points = []  # Liste pour les points du polygone
space_types = []     # Liste pour le type de la place de parking (0: Normal, 1: Handicapped)

# Chemins vers le fichier de positions et l'image de référence
position_file = r'C:/Users/Administrateur/Desktop/apprentissage/parking/Car_Parking_Space_Detector_YOLOv8-main/Space_ROIs.json'
image_path = r'C:/Users/Administrateur/Desktop/apprentissage/parking/Car_Parking_Space_Detector_YOLOv8-main/ROI_Reference.png'

# Charger les positions existantes à partir du fichier
posList = load_positions(position_file)

# Inspecter le contenu de posList pour vérifier sa structure
print("Contenu de posList chargé : ", posList)

# Fonction pour gérer les clics de souris
def mouseClick(event, x, y, flags, params):
    global polygon_points, posList, space_types

    # Ajouter un point lors du clic gauche
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))

        # Quand 4 points sont sélectionnés, demander le type de place
        if len(polygon_points) == 4:
            print("Select the space type: Press 'n' for normal, 'h' for handicapped")

    # Supprimer une zone en cliquant à droite sur une place
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, polygon in enumerate(posList):
            if cv2.pointPolygonTest(np.array(polygon['points'], dtype=np.int32), (x, y), False) >= 0:
                posList.pop(i)  # Supprimer le polygone sélectionné
                save_positions(position_file, posList)  # Sauvegarder les modifications
                break

# Lire l'image de référence
img = cv2.imread(image_path)

# Créer la fenêtre pour afficher l'image
cv2.imshow("Image", img)

# Configuration pour utiliser la fonction de clic de souris
cv2.setMouseCallback("Image", mouseClick)

while True:
    # Lire l'image de référence
    img = cv2.imread(image_path)
    
    # Dessiner les polygones des places de parking existantes
    for space in posList:
        # Vérification du format de chaque élément dans posList
        if isinstance(space, dict) and 'points' in space:
            pts = np.array(space['points'], np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if space['type'] == 1 else (0, 0, 255)  # Vert pour handicapé, Rouge pour normal
            cv2.polylines(img, [pts], True, color, 2)
        else:
            print("Erreur : Le format des données dans posList est incorrect", space)

    # Dessiner les points du polygone actuellement sélectionné
    for point in polygon_points:
        cv2.circle(img, point, 5, (0, 255, 0), -1)  # Dessiner des cercles verts pour les points

    # Afficher l'image
    cv2.imshow("Image", img)
    
    # Gestion des entrées clavier
    key = cv2.waitKey(1)

    if key == ord('q'):
        break  # Quitter si la touche 'q' est pressée
    elif key == ord('n') and len(polygon_points) == 4:  # Ajouter une place normale
        posList.append({'points': polygon_points.copy(), 'type': 0})
        save_positions(position_file, posList)  # Sauvegarder les positions
        polygon_points = []  # Réinitialiser les points pour un nouveau polygone
    elif key == ord('h') and len(polygon_points) == 4:  # Ajouter une place handicapée
        posList.append({'points': polygon_points.copy(), 'type': 1})
        save_positions(position_file, posList)  # Sauvegarder les positions
        polygon_points = []  # Réinitialiser les points pour un nouveau polygone

# Fermer toutes les fenêtres de OpenCV
cv2.destroyAllWindows()

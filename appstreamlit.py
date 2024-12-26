import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import json
import time
import os

# Définition de la fonction drawPolygons
def drawPolygons(frame, points_list, detection_points=None, polygon_colors=None, alpha=0.5):
    if polygon_colors is None:
        polygon_colors = {0: (30, 50, 180), 1: (30, 205, 50)}  # Couleurs pour normal et handicapé

    overlay = frame.copy()
    occupied_polygons = {0: 0, 1: 0}
    free_polygons = {0: 0, 1: 0}

    for area in points_list:
        area_np = np.array(area['points'], np.int32)
        color = polygon_colors[area['type']]
        is_inside = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points) if detection_points else False

        if is_inside:
            occupied_polygons[area['type']] += 1
        else:
            free_polygons[area['type']] += 1

        if area['type'] == 0:  # Normal place
            if is_inside:
                color = (0, 0, 139)  # Dark Red for occupied
            else:
                color = (0, 255, 0)  # Green for free
        elif area['type'] == 1:  # Handicap place
            if is_inside:
                color = (0, 165, 255)  # Orange for occupied
            else:
                color = (255, 0, 0)  # Blue for free

        cv2.fillPoly(overlay, [area_np], color)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame, occupied_polygons, free_polygons

# Définition de la fonction YOLO_Detection
def YOLO_Detection(model, frame, conf=0.35):
    results = model.predict(frame, conf=conf, classes=[0, 2])
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    return boxes, classes, names

# Fonction pour charger les positions depuis un fichier JSON
def load_positions_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Fonction pour dessiner la légende
def draw_legend(frame, occupied_counts, free_counts, total_counts, timers):
    legend_x = 20
    legend_y = 20
    legend_width = 250
    legend_height = 150

    # Création de la zone de légende transparente
    legend_overlay = frame.copy()
    cv2.rectangle(legend_overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(legend_overlay, 0.5, frame, 1 - 0.5, 0)

    # Texte pour la légende
    cv2.putText(frame, "Legend:", (legend_x + 10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Place handicapée libre (Bleu)
    cv2.putText(frame, f"Disabled space free: {free_counts[1]} / {total_counts[1]}", (legend_x + 10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Place normale libre (Verte)
    cv2.putText(frame, f"Normal place free: {free_counts[0]} / {total_counts[0]}", (legend_x + 10, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Afficher les chronomètres
    y_offset = 180
    for idx, timer in timers.items():
        if timer > 0:
            cv2.putText(frame, f"Place {idx} - Temps: {int(timer)}s", (legend_x + 10, legend_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
    
    return frame

# Initialisation du modèle YOLO
model = YOLO('C:/Users/Administrateur/Desktop/apprentissage/parking/Car_Parking_Space_Detector_YOLOv8-main/yolov8n.pt')  # Remplacez par le bon chemin du modèle YOLOv8

# Charger les positions des places de parking depuis un fichier JSON
position_file = r'C:/Users/Administrateur/Desktop/apprentissage/parking/Car_Parking_Space_Detector_YOLOv8-main/Space_ROIs.json'  # Remplacez par le bon chemin du fichier JSON
posList = load_positions_from_json(position_file)

# Initialiser les chronomètres pour chaque place
timers = {i: 0 for i in range(len(posList))}

# Créer une fonction pour lire et traiter la vidéo en temps réel
def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Créer un espace vide pour afficher l'image en temps réel
    frame_placeholder = st.empty()

    # Créer le fichier vidéo de sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détection d'objets avec YOLO
        boxes, classes, names = YOLO_Detection(model, frame)
        detection_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in boxes]

        # Dessiner les polygones et compter les places occupées et libres
        frame, occupied_counts, free_counts = drawPolygons(frame, posList, detection_points)

        # Mettre à jour les chronomètres
        for idx, area in enumerate(posList):
            area_np = np.array(area['points'], np.int32)
            is_inside = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points)
            if is_inside:
                timers[idx] += 1 / 30  # Ajoute 1 seconde (en supposant 30 fps)
            else:
                timers[idx] = 0

        # Calcul des totaux
        total_counts = {0: len([area for area in posList if area['type'] == 0]), 1: len([area for area in posList if area['type'] == 1])}

        # Ajouter la légende avec les chronomètres
        frame = draw_legend(frame, occupied_counts, free_counts, total_counts, timers)

        # Afficher l'image dans Streamlit de manière continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Écrire la frame dans le fichier vidéo de sortie
        out.write(frame)

        time.sleep(1 / 30)  # Simuler un FPS de 30 pour la vidéo en temps réel

    cap.release()
    out.release()

# Interface utilisateur Streamlit
st.title("🚗 Parking Space Detector with YOLOv8")

st.markdown("""
### "Data is the new oil, but a skilled data analyst with AI is the alchemist who turns it into gold... and occasionally programs the robot to do your job."  
- fotiosor 😊  
""")

# Chargement des fichiers vidéo et JSON
video_file = st.file_uploader("Charger la vidéo (mp4)", type=["mp4"])
json_file = st.file_uploader("Charger le fichier JSON des positions", type=["json"])

if video_file and json_file:
    # Sauvegarder les fichiers chargés localement
    video_path = "input_video.mp4"
    json_path = "Space_ROIs.json"

    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    with open(json_path, "wb") as f:
        f.write(json_file.getbuffer())

    # Charger les positions du fichier JSON
    posList = load_positions_from_json(json_path)

    # Créer un chemin pour le fichier de sortie
    output_video_path = "output_video.mp4"

    st.write("Traitement de la vidéo en cours...")
    process_video(video_path, output_video_path)
    
    # Option pour télécharger la vidéo traitée
    st.download_button("Télécharger la vidéo traitée", output_video_path, file_name="output_video.mp4")

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
from ultralytics import YOLO
from datetime import datetime
from PIL import Image

# Citation inspirante
st.title("🚗 AI-Powered Parking Space Detection")
st.markdown("""
> **"Data is the new oil, but a skilled data analyst with AI is the alchemist who turns it into gold...  
> and occasionally programs the robot to do your job."**  
> _— By F. F, with humor 😄_
""")

# Légende interactive
st.sidebar.markdown("### Legend")
st.sidebar.markdown("🟥 **Red**: Occupied parking space")
st.sidebar.markdown("🟩 **Green**: Free parking space")
st.sidebar.markdown("⏱️ **Timer**: Duration since last status change")

# Charger les fichiers
video_file = st.file_uploader("Upload a video file", type=["mp4"])
json_file = st.file_uploader("Upload a JSON file defining parking spaces", type=["json"])

if video_file and json_file:
    # Créer des fichiers temporaires
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
        tmp_json.write(json_file.read())
        json_path = tmp_json.name

    # Charger les positions des places
    def load_positions_from_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    pos_list = load_positions_from_json(json_path)

    # Initialiser les chronomètres pour chaque place
    timers = [{'start': None, 'last_status': 'Free'} for _ in range(len(pos_list))]

    # Charger le modèle YOLO
    model = YOLO("yolov8n.pt")

    # Lancer la détection en temps réel
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    # Sauvegarde de la vidéo annotée
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Détection YOLO
        def yolo_detection(model, frame, conf=0.35):
            results = model.predict(frame, conf=conf, classes=[0, 2], stream=True)
            detection_points = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    detection_points.append(((x1 + x2) // 2, (y1 + y2) // 2))
            return detection_points
        detection_points = yolo_detection(model, frame)

        # Dessiner les polygones et analyser l'occupation
        def draw_polygons(frame, points_list, detection_points=None, alpha=0.5, timers=None):
            overlay = frame.copy()
            current_time = datetime.now()

            for idx, area in enumerate(points_list):
                area_np = np.array(area['points'], np.int32)
                is_inside = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points) if detection_points else False

                # Mettre à jour les chronomètres
                if timers is not None:
                    if is_inside:
                        if timers[idx]['last_status'] == 'Free':
                            timers[idx]['start'] = current_time
                        timers[idx]['last_status'] = 'Occupied'
                    else:
                        if timers[idx]['last_status'] == 'Occupied':
                            timers[idx]['start'] = current_time
                        timers[idx]['last_status'] = 'Free'

                # Dessiner les polygones
                color = (0, 0, 255) if is_inside else (0, 255, 0)
                cv2.fillPoly(overlay, [area_np], color)

                # Ajouter le chronomètre
                if timers and timers[idx]['start']:
                    elapsed_time = current_time - timers[idx]['start']
                    elapsed_str = str(elapsed_time).split('.')[0]
                    centroid = np.mean(area_np, axis=0).astype(int)
                    cv2.putText(frame, elapsed_str, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        frame = draw_polygons(frame, pos_list, detection_points, timers=timers)

        # Sauvegarder la frame annotée
        out.write(frame)

        # Afficher la frame dans Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Mettre à jour la barre de progression
        progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count * 100)
        progress_bar.progress(progress)

    cap.release()
    out.release()

    st.success("✅ Video processing completed!")
    st.markdown("### Download the annotated video:")
    st.download_button("📥 Download Video", data=open(output_path, "rb").read(), file_name="annotated_video.mp4")

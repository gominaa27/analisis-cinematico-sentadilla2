import tempfile
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Análisis Kinemático de Sentadilla", page_icon="🏋️‍♂️", layout="centered"
)

st.title("Análisis cinemático de sentadilla")
st.write(
    "Esta aplicación permite analizar el ángulo de la rodilla durante la"
    " ejecución de una sentadilla mediante estimación de pose."
)

st.subheader("Selecciona el video de la sentadilla")

video_file = st.file_uploader(
    "Sube tu video aquí", type=["mp4", "mov", "avi", "webm"]
)

if video_file is not None:
  tfile = tempfile.NamedTemporaryFile(delete=False)
  tfile.write(video_file.read())

  st.video(video_file)

  if st.button("Analizar gesto", type="primary"):
    with st.spinner("Procesando fotogramas con MediaPipe..."):
      cap = cv2.VideoCapture(tfile.name)

      mp_pose = mp.solutions.pose
      pose = mp_pose.Pose(
          static_image_mode=False,
          model_complexity=1,
          enable_segmentation=False,
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5,
      )

      angulos_rodilla = []
      fps = cap.get(cv2.CAP_PROP_FPS)
      if fps == 0 or fps is None:
        fps = 30

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
          landmarks = results.pose_landmarks.landmark

          hip = np.array(
              [
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
              ]
          )
          knee = np.array(
              [
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
              ]
          )
          ankle = np.array(
              [
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
              ]
          )

          vector_a = hip - knee
          vector_b = ankle - knee

          cosine_angle = np.dot(vector_a, vector_b) / (
              np.linalg.norm(vector_a) * np.linalg.norm(vector_b) + 1e-6
          )
          angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
          angle_deg = np.degrees(angle)

          angulos_rodilla.append(angle_deg)

      cap.release()

    if angulos_rodilla:
      angulo_minimo = min(angulos_rodilla)
      st.success("¡Análisis completado con éxito!")
      st.metric(
          label="Ángulo Máximo de Flexión de Rodilla (Mínimo registrado)",
          value=f"{angulo_minimo:.1f}°",
      )
      st.subheader("Gráfico: Variación del ángulo de rodilla en el tiempo")
      st.line_chart(data=angulos_rodilla)
    else:
      st.warning(
          "No se pudieron detectar los puntos corporales en el video. Asegúrate"
          " de que la extremidad inferior analizada sea visible."
      )

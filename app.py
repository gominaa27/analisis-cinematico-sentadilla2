import tempfile

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st


# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------

st.set_page_config(
    page_title="Análisis Kinemático de Sentadilla",
    page_icon="🏋️‍♂️",
    layout="centered",
)

st.title("Análisis cinemático de sentadilla")

st.write(
    "Esta aplicación permite analizar el ángulo de la rodilla durante "
    "la ejecución de una sentadilla mediante estimación de pose."
)

st.subheader("Selecciona el video de la sentadilla")

video_file = st.file_uploader(
    "Sube tu video aquí",
    type=["mp4", "mov", "avi", "webm"],
)


# ---------------------------------------------------------
# CARGA DEL VIDEO
# ---------------------------------------------------------

if video_file is not None:

    # Crear archivo temporal para que OpenCV pueda leer el video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.close()

    # Mostrar video cargado
    st.video(video_file)

    # -----------------------------------------------------
    # BOTÓN DE ANÁLISIS
    # -----------------------------------------------------

    if st.button("Analizar gesto", type="primary"):

        with st.spinner("Procesando fotogramas con MediaPipe..."):

            # Abrir video
            cap = cv2.VideoCapture(tfile.name)

            # Inicializar MediaPipe Pose
            mp_pose = mp.solutions.pose

            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Lista donde se almacenarán los ángulos
            angulos_rodilla = []

            # Obtener FPS del video
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0 or fps is None:
                fps = 30

            # Obtener número total de fotogramas
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # Duración aproximada del video
            duracion = total_frames / fps if fps > 0 else 0

            # -------------------------------------------------
            # PROCESAMIENTO FOTOGRAMA A FOTOGRAMA
            # -------------------------------------------------

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                # Convertir de BGR a RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Estimar pose
                results = pose.process(image_rgb)

                # Si se detectan landmarks
                if results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark

                    # -----------------------------------------
                    # OBTENER CADERA, RODILLA Y TOBILLO DERECHOS
                    # -----------------------------------------

                    hip = np.array(
                        [
                            landmarks[
                                mp_pose.PoseLandmark.RIGHT_HIP.value
                            ].x,
                            landmarks[
                                mp_pose.PoseLandmark.RIGHT_HIP.value
                            ].y,
                        ]
                    )

                    knee = np.array(
                        [
                            landmarks[
                                mp_pose.PoseLandmark.RIGHT_KNEE.value
                            ].x,
                            landmarks[
                                mp_pose.PoseLandmark.RIGHT_KNEE.value
                            ].y,
                        ]
                    )

                    ankle = np.array(
                        [
                            landmarks[
                                mp_pose.PoseLandmark.RIGHT_ANKLE.value
                            ].x,
                            landmarks[
                                mp_pose.PoseLandmark.RIGHT_ANKLE.value
                            ].y,
                        ]
                    )

                    # -----------------------------------------
                    # CÁLCULO DEL ÁNGULO DE RODILLA
                    # -----------------------------------------

                    # Vectores desde la rodilla
                    vector_a = hip - knee
                    vector_b = ankle - knee

                    # Producto punto
                    denominador = (
                        np.linalg.norm(vector_a)
                        * np.linalg.norm(vector_b)
                    )

                    if denominador > 1e-6:

                        cosine_angle = np.dot(
                            vector_a,
                            vector_b,
                        ) / denominador

                        # Evitar errores numéricos
                        cosine_angle = np.clip(
                            cosine_angle,
                            -1.0,
                            1.0,
                        )

                        # Convertir de radianes a grados
                        angle = np.arccos(cosine_angle)
                        angle_deg = np.degrees(angle)

                        angulos_rodilla.append(angle_deg)

            # Liberar recursos
            cap.release()
            pose.close()

        # -----------------------------------------------------
        # RESULTADOS
        # -----------------------------------------------------

        if angulos_rodilla:

            # Menor ángulo = mayor flexión de rodilla
            angulo_maxima_flexion = min(angulos_rodilla)

            st.success("¡Análisis completado con éxito!")

            st.subheader("Resultados del análisis")

            # Mostrar resultados principales
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Máxima flexión de rodilla",
                    value=f"{angulo_maxima_flexion:.1f}°",
                )

            with col2:
                st.metric(
                    label="Duración del video",
                    value=f"{duracion:.2f} s",
                )

            # -------------------------------------------------
            # GRÁFICO
            # -------------------------------------------------

            st.subheader(
                "Variación del ángulo de rodilla durante la ejecución"
            )

            # Crear eje temporal
            tiempos = np.arange(len(angulos_rodilla)) / fps

            # Crear estructura para Streamlit
            datos_grafico = {
                "Tiempo (s)": tiempos,
                "Ángulo de rodilla (°)": angulos_rodilla,
            }

            st.line_chart(
                datos_grafico,
                x="Tiempo (s)",
                y="Ángulo de rodilla (°)",
            )

            # -------------------------------------------------
            # INTERPRETACIÓN
            # -------------------------------------------------

            st.info(
                "El valor de máxima flexión corresponde al menor ángulo "
                "de rodilla registrado durante la ejecución de la "
                "sentadilla."
            )

        else:

            st.warning(
                "No se pudieron detectar los puntos corporales en el video. "
                "Asegúrate de que el sujeto sea visible de cuerpo completo "
                "y que la extremidad inferior analizada permanezca visible "
                "durante la ejecución."
            )

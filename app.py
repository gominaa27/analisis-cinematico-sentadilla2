import tempfile
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Análisis cinemático de sentadilla",
    page_icon="🏋️",
    layout="centered"
)

st.title("Análisis cinemático de sentadilla")

st.write(
    "Esta aplicación permite analizar el ángulo de la rodilla "
    "durante la ejecución de una sentadilla mediante estimación de pose."
)

video_file = st.file_uploader(
    "Selecciona el video de la sentadilla",
    type=["mp4", "mov", "avi", "webm"]
)

if video_file is not None:

    st.subheader("Video seleccionado")
    st.video(video_file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.close()

    if st.button("Analizar gesto", type="primary"):

        with st.spinner("Procesando video..."):

            cap = cv2.VideoCapture(tfile.name)

            mp_pose = mp.solutions.pose

            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            angulos_rodilla = []
            tiempos = []

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0 or fps is None:
                fps = 30

            frame_count = 0

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps

                image_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = pose.process(image_rgb)

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark

                    # Puntos corporales utilizados para calcular
                    # el ángulo de la rodilla.
                    hip = np.array([
                        landmarks[
                            mp_pose.PoseLandmark.RIGHT_HIP.value
                        ].x,
                        landmarks[
                            mp_pose.PoseLandmark.RIGHT_HIP.value
                        ].y
                    ])

                    knee = np.array([
                        landmarks[
                            mp_pose.PoseLandmark.RIGHT_KNEE.value
                        ].x,
                        landmarks[
                            mp_pose.PoseLandmark.RIGHT_KNEE.value
                        ].y
                    ])

                    ankle = np.array([
                        landmarks[
                            mp_pose.PoseLandmark.RIGHT_ANKLE.value
                        ].x,
                        landmarks[
                            mp_pose.PoseLandmark.RIGHT_ANKLE.value
                        ].y
                    ])

                    vector_a = hip - knee
                    vector_b = ankle - knee

                    norm_a = np.linalg.norm(vector_a)
                    norm_b = np.linalg.norm(vector_b)

                    if norm_a > 0 and norm_b > 0:

                        cosine_angle = np.dot(
                            vector_a,
                            vector_b
                        ) / (norm_a * norm_b)

                        cosine_angle = np.clip(
                            cosine_angle,
                            -1.0,
                            1.0
                        )

                        angle = np.arccos(cosine_angle)
                        angle_deg = np.degrees(angle)

                        angulos_rodilla.append(angle_deg)
                        tiempos.append(current_time)

            cap.release()
            pose.close()

        if len(angulos_rodilla) > 0:

            st.success("¡Análisis completado correctamente!")

            angulo_minimo = min(angulos_rodilla)

            indice_minimo = angulos_rodilla.index(
                angulo_minimo
            )

            tiempo_mayor_flexion = tiempos[indice_minimo]

            st.subheader("Resultados principales")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Ángulo mínimo de rodilla",
                    f"{angulo_minimo:.1f}°"
                )

            with col2:
                st.metric(
                    "Tiempo de mayor flexión",
                    f"{tiempo_mayor_flexion:.2f} s"
                )

            st.write(
                "El ángulo mínimo corresponde al momento "
                "de mayor flexión de la rodilla durante "
                "la ejecución de la sentadilla."
            )

            st.subheader(
                "Variación del ángulo de rodilla durante la ejecución"
            )

            chart_data = {
                "Tiempo (s)": tiempos,
                "Ángulo de rodilla (°)": angulos_rodilla
            }

            st.line_chart(
                chart_data,
                x="Tiempo (s)",
                y="Ángulo de rodilla (°)"
            )

            st.subheader("Datos del análisis")

            cantidad = min(
                15,
                len(angulos_rodilla)
            )

            indices = np.linspace(
                0,
                len(angulos_rodilla) - 1,
                cantidad,
                dtype=int
            )

            tabla = []

            for i in indices:

                tabla.append({
                    "Tiempo (s)": round(
                        tiempos[i],
                        2
                    ),
                    "Ángulo (°)": round(
                        angulos_rodilla[i],
                        1
                    )
                })

            st.dataframe(
                tabla,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No fue posible detectar los puntos corporales "
                "necesarios para calcular el ángulo de rodilla. "
                "Verifica que la persona se encuentre de perfil, "
                "que la extremidad inferior analizada sea visible "
                "y que exista buena iluminación."
            )

import tempfile
import os

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
    "Esta aplicación permite analizar variables cinemáticas de la rodilla "
    "durante la ejecución de una sentadilla mediante estimación de pose."
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

    # Mostrar video original
    st.video(video_file)

    # -----------------------------------------------------
    # BOTÓN DE ANÁLISIS
    # -----------------------------------------------------

    if st.button("Analizar gesto", type="primary"):

        with st.spinner("Procesando fotogramas con MediaPipe..."):

            # ---------------------------------------------
            # ABRIR VIDEO
            # ---------------------------------------------

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

            # Obtener FPS
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0 or fps is None:
                fps = 30

            # Obtener dimensiones del video
            ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Obtener número total de fotogramas
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # Duración aproximada
            duracion = total_frames / fps if fps > 0 else 0

            # ---------------------------------------------
            # CREAR VIDEO PROCESADO
            # ---------------------------------------------

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_file.name
            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (ancho, alto),
            )

            # ---------------------------------------------
            # PROCESAMIENTO FOTOGRAMA A FOTOGRAMA
            # ---------------------------------------------

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                # Convertir de BGR a RGB
                image_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Estimar pose
                results = pose.process(image_rgb)

                # -----------------------------------------
                # SI SE DETECTA LA POSE
                # -----------------------------------------

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark

                    # -------------------------------------
                    # OBTENER LANDMARKS DERECHOS
                    # -------------------------------------

                    hip_landmark = landmarks[
                        mp_pose.PoseLandmark.RIGHT_HIP.value
                    ]

                    knee_landmark = landmarks[
                        mp_pose.PoseLandmark.RIGHT_KNEE.value
                    ]

                    ankle_landmark = landmarks[
                        mp_pose.PoseLandmark.RIGHT_ANKLE.value
                    ]

                    # -------------------------------------
                    # CONVERTIR COORDENADAS NORMALIZADAS
                    # A PIXELES
                    # -------------------------------------

                    hip_pixel = np.array(
                        [
                            int(hip_landmark.x * ancho),
                            int(hip_landmark.y * alto),
                        ]
                    )

                    knee_pixel = np.array(
                        [
                            int(knee_landmark.x * ancho),
                            int(knee_landmark.y * alto),
                        ]
                    )

                    ankle_pixel = np.array(
                        [
                            int(ankle_landmark.x * ancho),
                            int(ankle_landmark.y * alto),
                        ]
                    )

                    # -------------------------------------
                    # CALCULAR ÁNGULO DE RODILLA
                    # -------------------------------------

                    # Vectores desde la rodilla
                    vector_a = hip_pixel - knee_pixel
                    vector_b = ankle_pixel - knee_pixel

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

                        # Ángulo en radianes
                        angle = np.arccos(cosine_angle)

                        # Convertir a grados
                        angle_deg = np.degrees(angle)

                        # Guardar ángulo para los resultados
                        angulos_rodilla.append(angle_deg)

                        # ---------------------------------
                        # DIBUJAR PUNTOS
                        # ---------------------------------

                        cv2.circle(
                            frame,
                            tuple(hip_pixel),
                            8,
                            (0, 255, 0),
                            -1,
                        )

                        cv2.circle(
                            frame,
                            tuple(knee_pixel),
                            10,
                            (0, 0, 255),
                            -1,
                        )

                        cv2.circle(
                            frame,
                            tuple(ankle_pixel),
                            8,
                            (255, 0, 0),
                            -1,
                        )

                        # ---------------------------------
                        # DIBUJAR VECTORES
                        # ---------------------------------

                        cv2.line(
                            frame,
                            tuple(hip_pixel),
                            tuple(knee_pixel),
                            (0, 255, 255),
                            4,
                        )

                        cv2.line(
                            frame,
                            tuple(knee_pixel),
                            tuple(ankle_pixel),
                            (0, 255, 255),
                            4,
                        )

                        # ---------------------------------
                        # DIBUJAR ÁNGULO
                        # ---------------------------------

                        texto_angulo = f"{angle_deg:.1f} deg"

                        texto_x = knee_pixel[0] + 15
                        texto_y = knee_pixel[1] - 20

                        cv2.putText(
                            frame,
                            texto_angulo,
                            (
                                texto_x,
                                texto_y,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        # ---------------------------------
                        # ETIQUETAS
                        # ---------------------------------

                        cv2.putText(
                            frame,
                            "Cadera",
                            (
                                hip_pixel[0] + 10,
                                hip_pixel[1],
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.putText(
                            frame,
                            "Rodilla",
                            (
                                knee_pixel[0] + 10,
                                knee_pixel[1] + 25,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.putText(
                            frame,
                            "Tobillo",
                            (
                                ankle_pixel[0] + 10,
                                ankle_pixel[1],
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

                # -----------------------------------------
                # GUARDAR FOTOGRAMA PROCESADO
                # -----------------------------------------

                out.write(frame)

            # Liberar recursos
            cap.release()
            out.release()
            pose.close()

        # -------------------------------------------------
        # RESULTADOS
        # -------------------------------------------------

        if angulos_rodilla:

            # ---------------------------------------------
            # VARIABLES CINEMÁTICAS
            # ---------------------------------------------

            # Menor ángulo = mayor flexión de rodilla
            angulo_minimo = min(angulos_rodilla)

            # Mayor ángulo registrado
            angulo_maximo = max(angulos_rodilla)

            # Rango de movimiento (ROM)
            rom_rodilla = angulo_maximo - angulo_minimo

            # Índice del fotograma donde ocurre
            # la máxima flexión
            indice_maxima_flexion = angulos_rodilla.index(
                angulo_minimo
            )

            # Tiempo hasta la máxima flexión
            tiempo_maxima_flexion = (
                indice_maxima_flexion / fps
            )

            # ---------------------------------------------
            # MENSAJE DE ÉXITO
            # ---------------------------------------------

            st.success(
                "¡Análisis completado con éxito!"
            )

            # ---------------------------------------------
            # VIDEO PROCESADO
            # ---------------------------------------------

            st.subheader(
                "Visualización del análisis"
            )

            st.write(
                "Se muestran la cadera, rodilla y tobillo "
                "derechos, junto con los vectores utilizados "
                "para calcular el ángulo de la rodilla."
            )

            # Mostrar video procesado
            with open(
                output_path,
                "rb"
            ) as video_processed:

                st.video(
                    video_processed.read()
                )

            # ---------------------------------------------
            # RESULTADOS DEL ANÁLISIS
            # ---------------------------------------------

            st.subheader(
                "Resultados del análisis"
            )

            # ---------------------------------------------
            # MÉTRICAS PRINCIPALES
            # ---------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    label="Máxima flexión de rodilla",
                    value=f"{angulo_minimo:.1f}°",
                )

            with col2:

                st.metric(
                    label="ROM de rodilla",
                    value=f"{rom_rodilla:.1f}°",
                )

            with col3:

                st.metric(
                    label="Tiempo hasta máxima flexión",
                    value=f"{tiempo_maxima_flexion:.2f} s",
                )

            # Duración del video
            st.metric(
                label="Duración del video",
                value=f"{duracion:.2f} s",
            )

            # ---------------------------------------------
            # GRÁFICO
            # ---------------------------------------------

            st.subheader(
                "Variación del ángulo de rodilla durante la ejecución"
            )

            # Crear eje temporal
            tiempos = (
                np.arange(len(angulos_rodilla))
                / fps
            )

            # Crear estructura para el gráfico
            datos_grafico = {
                "Tiempo (s)": tiempos,
                "Ángulo de rodilla (°)": angulos_rodilla,
            }

            st.line_chart(
                datos_grafico,
                x="Tiempo (s)",
                y="Ángulo de rodilla (°)",
            )

            # ---------------------------------------------
            # TABLA RESUMIDA POR SEGUNDOS ENTEROS
            # ---------------------------------------------

            st.subheader(
                "Datos del análisis por segundo"
            )

            # Crear segundos enteros desde 0 hasta
            # el último segundo disponible
            segundos = np.arange(
                0,
                int(np.floor(duracion)) + 1,
                1,
            )

            # Guardar el ángulo correspondiente
            # a cada segundo
            angulos_por_segundo = []

            for segundo in segundos:

                # Buscar el fotograma más cercano
                # a ese segundo
                indice = int(
                    round(segundo * fps)
                )

                # Evitar superar los datos disponibles
                if indice >= len(angulos_rodilla):

                    indice = (
                        len(angulos_rodilla) - 1
                    )

                angulos_por_segundo.append(
                    angulos_rodilla[indice]
                )

            # Crear tabla
            datos_tabla = {

                "Tiempo (s)": segundos,

                "Ángulo de rodilla (°)": [
                    round(angulo, 1)
                    for angulo in angulos_por_segundo
                ],
            }

            st.dataframe(
                datos_tabla,
                hide_index=True,
                use_container_width=True,
            )

            # ---------------------------------------------
            # INTERPRETACIÓN
            # ---------------------------------------------

            st.info(
                "La máxima flexión de rodilla corresponde "
                "al menor ángulo registrado. El ROM representa "
                "la diferencia entre el mayor y el menor ángulo "
                "de rodilla durante el análisis."
            )

        else:

            st.warning(
                "No se pudieron detectar los puntos corporales "
                "en el video. Asegúrate de que el sujeto sea "
                "visible de cuerpo completo y que la extremidad "
                "inferior analizada permanezca visible durante "
                "la ejecución."
            )

        # ---------------------------------------------
        # ELIMINAR ARCHIVO TEMPORAL DEL VIDEO PROCESADO
        # ---------------------------------------------

        if os.path.exists(output_path):
            os.remove(output_path)

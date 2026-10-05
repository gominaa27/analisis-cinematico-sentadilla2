import os
import subprocess
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
    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4",
    )

    tfile.write(video_file.read())
    tfile.close()

    # Mostrar video original
    st.video(video_file)

    # -----------------------------------------------------
    # BOTÓN DE ANÁLISIS
    # -----------------------------------------------------

    if st.button("Analizar gesto", type="primary"):

        with st.spinner(
            "Procesando video y calculando variables cinemáticas..."
        ):

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

            # Lista para almacenar los ángulos
            angulos_rodilla = []

            # ---------------------------------------------
            # INFORMACIÓN DEL VIDEO
            # ---------------------------------------------

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0 or fps is None:
                fps = 30

            ancho = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            alto = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            total_frames = cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )

            duracion = (
                total_frames / fps
                if fps > 0
                else 0
            )

            # ---------------------------------------------
            # ARCHIVO TEMPORAL PARA VIDEO PROCESADO
            # ---------------------------------------------

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            output_path = output_file.name
            output_file.close()

            # Códec temporal para OpenCV
            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

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

                # -----------------------------------------
                # CONVERSIÓN BGR → RGB
                # -----------------------------------------

                image_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                # -----------------------------------------
                # ESTIMACIÓN DE POSE
                # -----------------------------------------

                results = pose.process(
                    image_rgb
                )

                # -----------------------------------------
                # SI SE DETECTA LA POSE
                # -----------------------------------------

                if results.pose_landmarks:

                    landmarks = (
                        results.pose_landmarks.landmark
                    )

                    # -------------------------------------
                    # LANDMARKS DERECHOS
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
                    # COORDENADAS EN PIXELES
                    # -------------------------------------

                    hip_pixel = np.array(
                        [
                            int(
                                hip_landmark.x
                                * ancho
                            ),
                            int(
                                hip_landmark.y
                                * alto
                            ),
                        ]
                    )

                    knee_pixel = np.array(
                        [
                            int(
                                knee_landmark.x
                                * ancho
                            ),
                            int(
                                knee_landmark.y
                                * alto
                            ),
                        ]
                    )

                    ankle_pixel = np.array(
                        [
                            int(
                                ankle_landmark.x
                                * ancho
                            ),
                            int(
                                ankle_landmark.y
                                * alto
                            ),
                        ]
                    )

                    # -------------------------------------
                    # VECTORES DESDE LA RODILLA
                    # -------------------------------------

                    vector_a = (
                        hip_pixel - knee_pixel
                    )

                    vector_b = (
                        ankle_pixel - knee_pixel
                    )

                    # -------------------------------------
                    # CÁLCULO DEL ÁNGULO
                    # -------------------------------------

                    denominador = (
                        np.linalg.norm(vector_a)
                        * np.linalg.norm(vector_b)
                    )

                    if denominador > 1e-6:

                        cosine_angle = (
                            np.dot(
                                vector_a,
                                vector_b,
                            )
                            / denominador
                        )

                        # Evitar errores numéricos
                        cosine_angle = np.clip(
                            cosine_angle,
                            -1.0,
                            1.0,
                        )

                        # Ángulo en radianes
                        angle_rad = np.arccos(
                            cosine_angle
                        )

                        # Convertir a grados
                        angle_deg = np.degrees(
                            angle_rad
                        )

                        # Guardar ángulo
                        angulos_rodilla.append(
                            angle_deg
                        )

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

                        texto_angulo = (
                            f"{angle_deg:.1f} deg"
                        )

                        texto_x = (
                            knee_pixel[0] + 15
                        )

                        texto_y = (
                            knee_pixel[1] - 20
                        )

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

            # ---------------------------------------------
            # LIBERAR RECURSOS
            # ---------------------------------------------

            cap.release()
            out.release()
            pose.close()

            # ---------------------------------------------
            # CONVERTIR A H.264 PARA REPRODUCCIÓN WEB
            # ---------------------------------------------

            video_web = tempfile.NamedTemporaryFile(
                delete=False,
                suffix="_web.mp4",
            )

            video_web_path = video_web.name
            video_web.close()

            resultado_ffmpeg = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    output_path,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    video_web_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

        # -------------------------------------------------
        # RESULTADOS
        # -------------------------------------------------

        if angulos_rodilla:

            # ---------------------------------------------
            # VARIABLES CINEMÁTICAS
            # ---------------------------------------------

            # Menor ángulo = mayor flexión
            angulo_minimo = min(
                angulos_rodilla
            )

            # Mayor ángulo registrado
            angulo_maximo = max(
                angulos_rodilla
            )

            # Rango de movimiento
            rom_rodilla = (
                angulo_maximo
                - angulo_minimo
            )

            # Índice de máxima flexión
            indice_maxima_flexion = (
                angulos_rodilla.index(
                    angulo_minimo
                )
            )

            # Tiempo hasta máxima flexión
            tiempo_maxima_flexion = (
                indice_maxima_flexion
                / fps
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
                "El video muestra la cadera, rodilla y "
                "tobillo derechos, junto con los vectores "
                "utilizados para calcular el ángulo de la rodilla."
            )

            # Verificar conversión
            if (
                resultado_ffmpeg.returncode == 0
                and os.path.exists(
                    video_web_path
                )
            ):

                with open(
                    video_web_path,
                    "rb",
                ) as video_processed:

                    st.video(
                        video_processed.read()
                    )

            else:

                st.warning(
                    "No fue posible generar el video "
                    "procesado para reproducción web."
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
                    value=(
                        f"{tiempo_maxima_flexion:.2f} s"
                    ),
                )

            # Duración
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

            tiempos = (
                np.arange(
                    len(angulos_rodilla)
                )
                / fps
            )

            datos_grafico = {
                "Tiempo (s)": tiempos,
                "Ángulo de rodilla (°)": (
                    angulos_rodilla
                ),
            }

            st.line_chart(
                datos_grafico,
                x="Tiempo (s)",
                y="Ángulo de rodilla (°)",
            )

            # ---------------------------------------------
            # TABLA POR SEGUNDOS ENTEROS
            # ---------------------------------------------

            st.subheader(
                "Datos del análisis por segundo"
            )

            segundos = np.arange(
                0,
                int(
                    np.floor(duracion)
                ) + 1,
                1,
            )

            angulos_por_segundo = []

            for segundo in segundos:

                indice = int(
                    round(
                        segundo * fps
                    )
                )

                if indice >= len(
                    angulos_rodilla
                ):

                    indice = (
                        len(
                            angulos_rodilla
                        )
                        - 1
                    )

                angulos_por_segundo.append(
                    angulos_rodilla[indice]
                )

            datos_tabla = {

                "Tiempo (s)": segundos,

                "Ángulo de rodilla (°)": [
                    round(
                        angulo,
                        1,
                    )
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

        # -------------------------------------------------
        # ELIMINAR ARCHIVOS TEMPORALES
        # -------------------------------------------------

        if os.path.exists(output_path):
            os.remove(output_path)

        if os.path.exists(video_web_path):
            os.remove(video_web_path)

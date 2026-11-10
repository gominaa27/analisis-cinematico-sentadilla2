
import os
import base64
import html
import subprocess
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Análisis cinemático de sentadilla",
    page_icon="🏋️",
    layout="centered"
)

st.title("Análisis cinemático de sentadilla")
st.write(
    "Carga un video de perfil para analizar el ángulo "
    "de rodilla durante una sentadilla."
)


def calcular_angulo(cadera, rodilla, tobillo):
    a = cadera - rodilla
    b = tobillo - rodilla
    norma = np.linalg.norm(a) * np.linalg.norm(b)

    if norma < 1e-6:
        return None

    coseno = np.clip(np.dot(a, b) / norma, -1, 1)
    return float(np.degrees(np.arccos(coseno)))


def convertir_h264(entrada, salida):
    comando = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", entrada,
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        salida,
    ]

    try:
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=180
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)

    valido = (
        resultado.returncode == 0
        and os.path.exists(salida)
        and os.path.getsize(salida) > 0
    )

    return valido, resultado.stderr[-1500:] if not valido else ""


def mostrar_video(datos_video):
    """Mostrar video directamente en el navegador."""
    b64 = base64.b64encode(datos_video).decode("ascii")

    st.markdown(
        f'<video controls playsinline preload="metadata" '
        f'style="width:100%;max-height:520px;'
        f'background:#111;border-radius:8px">'
        f'<source src="data:video/mp4;base64,{b64}" '
        f'type="video/mp4">'
        f'Tu navegador no admite este video.'
        f'</video>',
        unsafe_allow_html=True
    )


def enlace_descarga(datos_archivo, nombre, mime, etiqueta):
    """Descargar archivos pequeños directamente."""
    b64 = base64.b64encode(datos_archivo).decode("ascii")

    st.markdown(
        f'<a href="data:{mime};base64,{b64}" '
        f'download="{html.escape(nombre)}" '
        f'style="display:inline-block;padding:10px 15px;'
        f'border:1px solid #999;border-radius:8px;'
        f'text-decoration:none">'
        f'{html.escape(etiqueta)}</a>',
        unsafe_allow_html=True
    )


archivo = st.file_uploader(
    "Selecciona el video de la sentadilla",
    type=["mp4", "mov", "avi", "webm"]
)

if archivo is not None:

    lado = st.selectbox(
        "Pierna que se analizará",
        ["Derecha", "Izquierda"]
    )

    datos = archivo.getvalue()

    if len(datos) > 15 * 1024 * 1024:
        st.warning(
            "Para esta prueba utiliza un video de menos de 15 MB."
        )

    st.caption(
        f"Archivo recibido: {len(datos) / 1024 / 1024:.2f} MB"
    )

    with tempfile.TemporaryDirectory() as carpeta:

        extension = os.path.splitext(archivo.name)[1].lower() or ".mp4"

        ruta_original = os.path.join(
            carpeta, "entrada" + extension
        )

        ruta_web_original = os.path.join(
            carpeta, "original_web.mp4"
        )

        ruta_procesada = os.path.join(
            carpeta, "procesado_temporal.mp4"
        )

        ruta_web_procesada = os.path.join(
            carpeta, "analizado_web.mp4"
        )

        with open(ruta_original, "wb") as f:
            f.write(datos)

        # FOTOGRAMA DE COMPROBACIÓN

        st.subheader("Fotograma de comprobación")

        captura = cv2.VideoCapture(ruta_original)
        correcto, fotograma = captura.read()
        captura.release()

        if correcto:

            ok_jpg, buffer_jpg = cv2.imencode(
                ".jpg", fotograma
            )

            if ok_jpg:

                imagen_b64 = base64.b64encode(
                    buffer_jpg.tobytes()
                ).decode("ascii")

                st.markdown(
                    f'<img src="data:image/jpeg;base64,{imagen_b64}" '
                    f'style="width:100%;max-width:600px;'
                    f'border-radius:8px" '
                    f'alt="Fotograma de comprobación">',
                    unsafe_allow_html=True
                )

                st.caption("Imagen extraída del video")

            else:
                st.warning(
                    "No se pudo codificar el fotograma."
                )

        else:
            st.warning(
                "OpenCV no pudo extraer una imagen del video."
            )

        # VIDEO ORIGINAL

        st.subheader("Video original")

        with st.spinner("Preparando video compatible..."):
            original_ok, original_error = convertir_h264(
                ruta_original,
                ruta_web_original
            )

        if original_ok:

            with open(ruta_web_original, "rb") as f:
                original_web_bytes = f.read()

            mostrar_video(original_web_bytes)

            enlace_descarga(
                original_web_bytes,
                "sentadilla_original_h264.mp4",
                "video/mp4",
                "Descargar video original compatible"
            )

        else:
            st.warning(
                "No se pudo preparar la vista previa del video original."
            )
            st.code(original_error or "Error desconocido")

        # ANÁLISIS DE MOVIMIENTO

        if st.button("Analizar gesto", type="primary"):

            with st.spinner("Analizando la sentadilla..."):

                cap = cv2.VideoCapture(ruta_original)

                if not cap.isOpened():

                    st.error(
                        "No se pudo abrir el video para analizarlo."
                    )

                else:

                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if not np.isfinite(fps) or fps <= 0:
                        fps = 30.0

                    ancho = int(
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    )

                    alto = int(
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )

                    if ancho <= 0 or alto <= 0:

                        st.error(
                            "No se pudo leer la resolución del video."
                        )

                        cap.release()

                    else:

                        escritor = cv2.VideoWriter(
                            ruta_procesada,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (ancho, alto)
                        )

                        if not escritor.isOpened():

                            st.error(
                                "No se pudo crear el video analizado."
                            )

                            cap.release()

                        else:

                            mp_pose = mp.solutions.pose

                            if lado == "Derecha":

                                ids = (
                                    mp_pose.PoseLandmark.RIGHT_HIP.value,
                                    mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                    mp_pose.PoseLandmark.RIGHT_ANKLE.value
                                )

                            else:

                                ids = (
                                    mp_pose.PoseLandmark.LEFT_HIP.value,
                                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                                    mp_pose.PoseLandmark.LEFT_ANKLE.value
                                )

                            angulos = []
                            tiempos = []
                            n = 0

                            with mp_pose.Pose(
                                static_image_mode=False,
                                model_complexity=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5
                            ) as pose:

                                while True:

                                    ok, frame = cap.read()

                                    if not ok:
                                        break

                                    tiempo = n / fps
                                    n += 1

                                    rgb = cv2.cvtColor(
                                        frame,
                                        cv2.COLOR_BGR2RGB
                                    )

                                    resultado = pose.process(rgb)

                                    if resultado.pose_landmarks:

                                        seleccion = [
                                            resultado.pose_landmarks.landmark[i]
                                            for i in ids
                                        ]

                                        if all(
                                            p.visibility >= 0.5
                                            for p in seleccion
                                        ):

                                            puntos = [
                                                np.array(
                                                    [p.x * ancho, p.y * alto],
                                                    dtype=float
                                                )
                                                for p in seleccion
                                            ]

                                            angulo = calcular_angulo(
                                                *puntos
                                            )

                                            if angulo is not None:

                                                angulos.append(angulo)
                                                tiempos.append(tiempo)

                                                p1, p2, p3 = [
                                                    tuple(
                                                        np.rint(p).astype(int)
                                                    )
                                                    for p in puntos
                                                ]

                                                cv2.line(
                                                    frame, p1, p2,
                                                    (0, 255, 255), 4
                                                )

                                                cv2.line(
                                                    frame, p2, p3,
                                                    (0, 255, 255), 4
                                                )

                                                # Puntos con colores diferentes (BGR)
                                                etiquetas = [
                                                    (p1, "Cadera", (0, 255, 0)),
                                                    (p2, "Rodilla", (0, 0, 255)),
                                                    (p3, "Tobillo", (255, 0, 0)),
                                                ]

                                                escala = max(0.55, min(ancho / 600, 1.0))
                                                grosor = max(1, round(2 * escala))
                                                for punto, nombre, color in etiquetas:
                                                    cv2.circle(frame, punto, 8, color, -1)
                                                    texto_x = min(punto[0] + 12, ancho - 100)
                                                    texto_y = max(25, min(punto[1] + 22, alto - 10))
                                                    cv2.putText(
                                                        frame, nombre,
                                                        (max(5, texto_x), texto_y),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        escala, color, grosor,
                                                        cv2.LINE_AA
                                                    )

                                                # Angulo junto a la rodilla
                                                texto_angulo = f"{angulo:.1f} deg"
                                                x = max(5, min(p2[0] + 15, ancho - 160))
                                                y = max(30, min(p2[1] - 20, alto - 10))
                                                cv2.putText(
                                                    frame, texto_angulo, (x, y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    max(0.7, escala),
                                                    (0, 255, 255), 2, cv2.LINE_AA
                                                )

                                    escritor.write(frame)

                            cap.release()
                            escritor.release()

                            if n == 0:

                                st.error(
                                    "El video no contiene fotogramas legibles."
                                )

                            else:

                                procesado_ok, procesado_error = convertir_h264(
                                    ruta_procesada,
                                    ruta_web_procesada
                                )

                                if procesado_ok:

                                    with open(
                                        ruta_web_procesada, "rb"
                                    ) as f:
                                        procesado_bytes = f.read()

                                    st.subheader(
                                        "Video con análisis de movimiento"
                                    )

                                    mostrar_video(procesado_bytes)

                                    enlace_descarga(
                                        procesado_bytes,
                                        "sentadilla_analizada_h264.mp4",
                                        "video/mp4",
                                        "Descargar video analizado"
                                    )

                                else:

                                    st.warning(
                                        "No se pudo convertir el video analizado a H.264."
                                    )

                                    st.code(
                                        procesado_error or "Error desconocido"
                                    )

                                # RESULTADOS

                                if angulos:

                                    i_min = int(np.argmin(angulos))
                                    minimo = angulos[i_min]
                                    rom = max(angulos) - minimo

                                    st.success(
                                        "¡Análisis completado!"
                                    )

                                    st.subheader(
                                        "Resultados del análisis"
                                    )

                                    c1, c2, c3 = st.columns(3)

                                    c1.metric(
                                        "Ángulo mínimo",
                                        f"{minimo:.1f}°"
                                    )

                                    c2.metric(
                                        "ROM observado",
                                        f"{rom:.1f}°"
                                    )

                                    c3.metric(
                                        "Mayor flexión en",
                                        f"{tiempos[i_min]:.2f} s"
                                    )

                                    tabla = pd.DataFrame({
                                        "Tiempo (s)": tiempos,
                                        "Ángulo de rodilla (°)": angulos
                                    })

                                    st.subheader(
                                        "Ángulo de rodilla a lo largo del tiempo"
                                    )

                                    st.line_chart(
                                        tabla.set_index("Tiempo (s)")
                                    )

                                    st.subheader(
                                        "Datos del análisis"
                                    )

                                    st.dataframe(
                                        tabla.iloc[
                                            ::max(1, len(tabla) // 20)
                                        ].round(2),
                                        hide_index=True,
                                        use_container_width=True
                                    )

                                    st.caption(
                                        "Medición bidimensional aproximada "
                                        "basada en landmarks de MediaPipe; "
                                        "no equivale a una medición clínica validada."
                                    )

                                else:

                                    st.warning(
                                        "No se detectaron suficientes puntos "
                                        "para calcular el ángulo."
                                    )

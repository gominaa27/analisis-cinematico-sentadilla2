import os
import subprocess
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Análisis cinemático de sentadilla", page_icon="🏋️", layout="centered")
st.title("Análisis cinemático de sentadilla")
st.write("Carga un video de perfil para analizar el ángulo de rodilla durante una sentadilla.")


def convertir_a_web(origen, destino):
    comando = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", origen,
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", destino,
    ]
    try:
        proceso = subprocess.run(comando, capture_output=True, text=True, timeout=180)
        if proceso.returncode != 0 or not os.path.exists(destino) or os.path.getsize(destino) == 0:
            return False, proceso.stderr[-1500:] or "No se generó el archivo de video."
        return True, ""
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def calcular_angulo(cadera, rodilla, tobillo):
    a = cadera - rodilla
    b = tobillo - rodilla
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    if denominador < 1e-6:
        return None
    coseno = np.clip(np.dot(a, b) / denominador, -1.0, 1.0)
    return float(np.degrees(np.arccos(coseno)))


archivo = st.file_uploader("Selecciona el video de la sentadilla", type=["mp4", "mov", "avi", "webm"])

if archivo is not None:
    lado = st.selectbox("Pierna que se analizará", ["Derecha", "Izquierda"])
    with tempfile.TemporaryDirectory() as carpeta:
        extension = os.path.splitext(archivo.name)[1].lower() or ".mp4"
        ruta_original = os.path.join(carpeta, "entrada" + extension)
        ruta_web = os.path.join(carpeta, "original_web.mp4")
        with open(ruta_original, "wb") as f:
            f.write(archivo.getvalue())

        st.subheader("Video original")
        ok_original, error_original = convertir_a_web(ruta_original, ruta_web)
        if ok_original:
            with open(ruta_web, "rb") as f:
                st.video(f.read(), format="video/mp4")
        else:
            st.warning("No se pudo convertir el video original para reproducirlo.")
            st.code(error_original)

        if st.button("Analizar gesto", type="primary"):
            with st.spinner("Analizando el movimiento..."):
                entrada = cv2.VideoCapture(ruta_original)
                if not entrada.isOpened():
                    st.error("OpenCV no pudo abrir el video. Prueba con un MP4 H.264.")
                else:
                    fps = entrada.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps <= 0:
                        fps = 30.0
                    ancho = int(entrada.get(cv2.CAP_PROP_FRAME_WIDTH))
                    alto = int(entrada.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if ancho <= 0 or alto <= 0:
                        st.error("No se pudieron leer las dimensiones del video.")
                        entrada.release()
                    else:
                        ruta_procesada = os.path.join(carpeta, "procesado_temporal.mp4")
                        ruta_procesada_web = os.path.join(carpeta, "procesado_web.mp4")
                        escritor = cv2.VideoWriter(
                            ruta_procesada, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
                        )
                        if not escritor.isOpened():
                            st.error("No se pudo crear el video procesado.")
                            entrada.release()
                        else:
                            mp_pose = mp.solutions.pose
                            indices = (
                                (mp_pose.PoseLandmark.RIGHT_HIP.value,
                                 mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                 mp_pose.PoseLandmark.RIGHT_ANKLE.value)
                                if lado == "Derecha" else
                                (mp_pose.PoseLandmark.LEFT_HIP.value,
                                 mp_pose.PoseLandmark.LEFT_KNEE.value,
                                 mp_pose.PoseLandmark.LEFT_ANKLE.value)
                            )
                            angulos = []
                            tiempos = []
                            numero_fotograma = 0
                            with mp_pose.Pose(
                                static_image_mode=False,
                                model_complexity=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5,
                            ) as pose:
                                while True:
                                    correcto, fotograma = entrada.read()
                                    if not correcto:
                                        break
                                    tiempo = numero_fotograma / fps
                                    numero_fotograma += 1
                                    rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
                                    resultado = pose.process(rgb)
                                    if resultado.pose_landmarks:
                                        puntos = resultado.pose_landmarks.landmark
                                        seleccion = [puntos[i] for i in indices]
                                        if all(p.visibility >= 0.5 for p in seleccion):
                                            pixeles = [
                                                np.array([int(p.x * ancho), int(p.y * alto)])
                                                for p in seleccion
                                            ]
                                            cadera, rodilla, tobillo = pixeles
                                            angulo = calcular_angulo(cadera, rodilla, tobillo)
                                            if angulo is not None:
                                                angulos.append(angulo)
                                                tiempos.append(tiempo)
                                                cv2.line(fotograma, tuple(cadera), tuple(rodilla), (0, 255, 255), 4)
                                                cv2.line(fotograma, tuple(rodilla), tuple(tobillo), (0, 255, 255), 4)
                                                for punto, color in zip(
                                                    pixeles, [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
                                                ):
                                                    cv2.circle(fotograma, tuple(punto), 8, color, -1)
                                                x = max(5, min(int(rodilla[0]) + 15, ancho - 120))
                                                y = max(25, min(int(rodilla[1]) - 15, alto - 10))
                                                cv2.putText(
                                                    fotograma, f"{angulo:.1f} deg", (x, y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3,
                                                    cv2.LINE_AA,
                                                )
                                                cv2.putText(
                                                    fotograma, f"{angulo:.1f} deg", (x, y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                                    cv2.LINE_AA,
                                                )
                                    escritor.write(fotograma)
                            entrada.release()
                            escritor.release()
                            if numero_fotograma == 0:
                                st.error("El video no contiene fotogramas que se puedan leer.")
                            else:
                                st.subheader("Video con análisis de movimiento")
                                ok_procesado, error_procesado = convertir_a_web(
                                    ruta_procesada, ruta_procesada_web
                                )
                                if ok_procesado:
                                    with open(ruta_procesada_web, "rb") as f:
                                        st.video(f.read(), format="video/mp4")
                                else:
                                    st.error("No fue posible convertir el video analizado para el navegador.")
                                    st.code(error_procesado)
                                if angulos:
                                    indice = int(np.argmin(angulos))
                                    minimo = angulos[indice]
                                    rom = max(angulos) - minimo
                                    st.success("¡Análisis completado!")
                                    st.subheader("Resultados")
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Ángulo mínimo", f"{minimo:.1f}°")
                                    c2.metric("ROM observado", f"{rom:.1f}°")
                                    c3.metric("Mayor flexión", f"{tiempos[indice]:.2f} s")
                                    tabla = pd.DataFrame({
                                        "Tiempo (s)": tiempos,
                                        "Ángulo de rodilla (°)": angulos,
                                    })
                                    st.subheader("Ángulo de rodilla a lo largo del tiempo")
                                    st.line_chart(tabla.set_index("Tiempo (s)"))
                                    st.subheader("Datos del análisis")
                                    paso = max(1, len(tabla) // 20)
                                    st.dataframe(
                                        tabla.iloc[::paso].round(2),
                                        hide_index=True, use_container_width=True,
                                    )
                                    st.caption(
                                        "El ángulo mínimo representa la mayor flexión observada. "
                                        "El ROM es la diferencia entre los ángulos máximo y mínimo medidos. "
                                        "Son estimaciones bidimensionales dependientes de la visibilidad y la cámara."
                                    )
                                else:
                                    st.warning(
                                        "No se detectaron suficientes puntos de cadera, rodilla y tobillo. "
                                        "Graba de perfil, con el cuerpo completo visible, y prueba la otra pierna."
                                    )

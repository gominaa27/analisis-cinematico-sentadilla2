import base64
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Análisis cinemático de sentadilla", page_icon="🏋️", layout="centered")
st.title("Análisis cinemático de sentadilla")
st.write("Carga un video de perfil para analizar el ángulo de rodilla durante una sentadilla.")

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
    
    tfile_orig = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile_orig.write(archivo.read())
    ruta_original = tfile_orig.name
    tfile_orig.close()

    st.subheader("Video original")
    st.video(ruta_original)
    
    with open(ruta_original, "rb") as f:
        video_orig_bytes = f.read()
    st.download_button("Descargar video original", video_orig_bytes, "video_original.mp4", "video/mp4")

    if st.button("Analizar gesto", type="primary"):
        with st.spinner("Procesando video y generando salida visual... ⏳"):
            entrada = cv2.VideoCapture(ruta_original)
            if not entrada.isOpened():
                st.error("OpenCV no pudo abrir el video.")
            else:
                fps = entrada.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0:
                    fps = 30.0
                ancho = int(entrada.get(cv2.CAP_PROP_FRAME_WIDTH))
                alto = int(entrada.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Archivo temporal para la salida procesada con códec estándar
                tfile_proc = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                ruta_procesada = tfile_proc.name
                tfile_proc.close()

                escritor = cv2.VideoWriter(
                    ruta_procesada, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto)
                )

                if not escritor.isOpened():
                    st.error("No se pudo crear el escritor de video.")
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
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3,
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
                                
                                if all(p.visibility >= 0.3 for p in seleccion):
                                    pixeles = [
                                        np.array([int(p.x * ancho), int(p.y * alto)])
                                        for p in seleccion
                                    ]
                                    cadera, rodilla, tobillo = pixeles
                                    angulo_actual = calcular_angulo(cadera, rodilla, tobillo)
                                    
                                    if angulo_actual is not None:
                                        angulos.append(angulo_actual)
                                        tiempos.append(tiempo)
                                        
                                        # Dibujar líneas amarillas y puntos
                                        cv2.line(fotograma, tuple(cadera), tuple(rodilla), (0, 255, 255), 4)
                                        cv2.line(fotograma, tuple(rodilla), tuple(tobillo), (0, 255, 255), 4)
                                        for punto, color in zip(
                                            pixeles, [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
                                        ):
                                            cv2.circle(fotograma, tuple(punto), 8, color, -1)
                                            
                                        x = max(5, min(int(rodilla[0]) + 15, ancho - 120))
                                        y = max(25, min(int(rodilla[1]) - 15, alto - 10))
                                        cv2.putText(
                                            fotograma, f"{angulo_actual:.1f} deg", (x, y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3,
                                            cv2.LINE_AA,
                                        )
                                        cv2.putText(
                                            fotograma, f"{angulo_actual:.1f} deg", (x, y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                            cv2.LINE_AA,
                                        )
                            
                            escritor.write(fotograma)
                            
                    entrada.release()
                    escritor.release()

                    if numero_fotograma == 0:
                        st.error("El video no contiene fotogramas.")
                    else:
                        st.subheader("Video con análisis de movimiento")
                        
                        # TRUCO HTML5 BASE64: Inyecta el video directo como código binario al navegador
                        with open(ruta_procesada, "rb") as f_proc:
                            video_bytes = f_proc.read()
                        
                        base64_video = base64.b64encode(video_bytes).decode("utf-8")
                        st.markdown(
                            f'<video width="100%" controls autoplay loop><source src="data:video/mp4;base64,{base64_video}" type="video/mp4"></video>',
                            unsafe_allow_html=True,
                        )
                        
                        st.download_button("Descargar video analizado", video_bytes, "sentadilla_analizada.mp4", "video/mp4")
                        
                        if angulos:
                            indice_min = int(np.argmin(angulos))
                            minimo = angulos[indice_min]
                            rom = max(angulos) - minimo
                            
                            st.success("¡Análisis completado con éxito!")
                            st.subheader("Resultados del análisis")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Máxima flexión de rodilla", f"{minimo:.1f}°")
                            c2.metric("ROM de rodilla", f"{rom:.1f}°")
                            c3.metric("Tiempo hasta máxima flexión", f"{tiempos[indice_min]:.2f} s")
                            
                            tabla = pd.DataFrame({
                                "Tiempo (s)": tiempos,
                                "Ángulo de rodilla (°)": angulos,
                            })
                            st.subheader("Variación del ángulo de rodilla durante la ejecución")
                            st.line_chart(tabla.set_index("Tiempo (s)"))
                            
                            st.subheader("Datos del análisis por segundo")
                            paso = max(1, len(tabla) // 20)
                            st.dataframe(
                                tabla.iloc[::paso].round(2),
                                hide_index=True, use_container_width=True,
                            )
                        else:
                            st.warning("No se pudieron calcular los ángulos. Asegúrate de grabar de perfil.")

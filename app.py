import tempfile
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Análisis cinemático de sentadilla", page_icon="🏋️", layout="centered")
st.title("Análisis cinemático de sentadilla")
st.write("Depuración de errores de carga y procesamiento.")

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

    if st.button("Analizar gesto", type="primary"):
        with st.spinner("Leyendo video con OpenCV..."):
            entrada = cv2.VideoCapture(ruta_original)
            
        if not entrada.isOpened():
            st.error("❌ ERROR CRÍTICO: OpenCV no pudo abrir el archivo de video. El formato no es compatible.")
        else:
            fps = entrada.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            total_frames = int(entrada.get(cv2.CAP_PROP_FRAME_COUNT))
            ancho = int(entrada.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(entrada.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            st.info(f"ℹ️ Información del video: {total_frames} fotogramas, Resolución: {ancho}x{alto}, FPS: {fps}")
            
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
            fotogramas_guardados = []
            numero_fotograma = 0
            frames_con_deteccion = 0
            
            with st.spinner("Procesando fotogramas con MediaPipe..."):
                with mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2,
                ) as pose:
                    while True:
                        correcto, fotograma = entrada.read()
                        if not correcto:
                            break
                        
                        tiempo = numero_fotograma / fps
                        numero_fotograma += 1
                        
                        rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
                        resultado = pose.process(rgb)
                        
                        frame_dibujado = fotograma.copy()
                        
                        if resultado.pose_landmarks:
                            puntos = resultado.pose_landmarks.landmark
                            seleccion = [puntos[i] for i in indices]
                            
                            # Relajamos la visibilidad al mínimo para descartar que sea por exigencia de MediaPipe
                            if all(p.visibility >= 0.1 for p in seleccion):
                                frames_con_deteccion += 1
                                pixeles = [
                                    np.array([int(p.x * ancho), int(p.y * alto)])
                                    for p in seleccion
                                ]
                                cadera, rodilla, tobillo = pixeles
                                angulo_actual = calcular_angulo(cadera, rodilla, tobillo)
                                
                                if angulo_actual is not None:
                                    angulos.append(angulo_actual)
                                    tiempos.append(tiempo)
                                    
                                    cv2.line(frame_dibujado, tuple(cadera), tuple(rodilla), (0, 255, 255), 4)
                                    cv2.line(frame_dibujado, tuple(rodilla), tuple(tobillo), (0, 255, 255), 4)
                                    for punto, color in zip(
                                        pixeles, [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
                                    ):
                                        cv2.circle(frame_dibujado, tuple(punto), 8, color, -1)
                                        
                                    x = max(5, min(int(rodilla[0]) + 15, ancho - 120))
                                    y = max(25, min(int(rodilla[1]) - 15, alto - 10))
                                    cv2.putText(
                                        frame_dibujado, f"{angulo_actual:.1f} deg", (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3,
                                        cv2.LINE_AA,
                                    )
                        
                        if numero_fotograma % 30 == 0:
                            fotogramas_guardados.append(cv2.cvtColor(frame_dibujado, cv2.COLOR_BGR2RGB))
                            
                    entrada.release()

            st.write(f"🔍 **Reporte de depuración:**")
            st.write(f"- Fotogramas totales leídos: {numero_fotograma}")
            st.write(f"- Fotogramas con detección de extremidades: {frames_con_deteccion}")
            st.write(f"- Ángulos calculados con éxito: {len(angulos)}")

            if numero_fotograma == 0:
                st.error("El video no contiene fotogramas legibles.")
            elif frames_con_deteccion == 0:
                st.error("⚠️ MediaPipe no detectó los puntos de la pierna seleccionada en ningún fotograma. Prueba cambiando de pierna (Derecha/Izquierda) o sube un video donde se vea claramente tu cuerpo entero de perfil.")
            elif not angulos:
                st.error("Se detectó el cuerpo pero no se pudo calcular el ángulo de la rodilla.")
            else:
                st.success("¡Análisis completado con éxito!")
                indice = int(np.argmin(angulos))
                minimo = angulos[indice]
                rom = max(angulos) - minimo
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Ángulo mínimo", f"{minimo:.1f}°")
                c2.metric("ROM observado", f"{rom:.1f}°")
                c3.metric("Mayor flexión", f"{tiempos[indice]:.2f} s")
                
                if fotogramas_guardados:
                    st.image(fotogramas_guardados[0], caption="Fotograma de muestra analizado", use_container_width=True)

                tabla = pd.DataFrame({
                    "Tiempo (s)": tiempos,
                    "Ángulo de rodilla (°)": angulos,
                })
                st.line_chart(tabla.set_index("Tiempo (s)"))

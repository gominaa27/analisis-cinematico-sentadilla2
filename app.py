if st.button("Analizar gesto", type="primary"):
            with st.spinner("Procesando video y dibujando el análisis..."):
                captura = cv2.VideoCapture(ruta_original)
                if not captura.isOpened():
                    st.error("OpenCV no pudo abrir el video. Prueba con un MP4 H.264.")
                else:
                    fps = captura.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps <= 0:
                        fps = 30.0
                    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
                    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if ancho <= 0 or alto <= 0:
                        st.error("No se pudieron leer las dimensiones del video.")
                        captura.release()
                    else:
                        ruta_temporal = os.path.join(carpeta, "analisis_temporal.mp4")
                        ruta_procesada = os.path.join(carpeta, "analisis_web.mp4")
                        escritor = cv2.VideoWriter(
                            ruta_temporal, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
                        )
                        tiempos = []
                        angulos = []
                        contador = 0
                        mp_pose = mp.solutions.pose
                        lado = st.session_state.get("lado_rodilla", "Derecha")
                        if lado == "Izquierda":
                            puntos = (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)
                        else:
                            puntos = (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)

                        with mp_pose.Pose(
                            static_image_mode=False,
                            model_complexity=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                        ) as pose:
                            while True:
                                leido, fotograma = captura.read()
                                if not leido:
                                    break
                                instante = contador / fps
                                contador += 1
                                rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
                                resultado = pose.process(rgb)
                                if resultado.pose_landmarks:
                                    marcas = resultado.pose_landmarks.landmark
                                    seleccion = [marcas[p.value] for p in puntos]
                                    # Evita medir puntos que MediaPipe no ve con suficiente claridad.
                                    if all(m.visibility >= 0.5 for m in seleccion):
                                        coordenadas = [np.array([m.x * ancho, m.y * alto], dtype=float) for m in seleccion]
                                        angulo = calcular_angulo(*coordenadas)
                                        if angulo is not None:
                                            tiempos.append(instante)
                                            angulos.append(angulo)
                                            cadera, rodilla, tobillo = [tuple(np.rint(p).astype(int)) for p in coordenadas]
                                            cv2.line(fotograma, cadera, rodilla, (0, 255, 255), 4)
                                            cv2.line(fotograma, rodilla, tobillo, (0, 255, 255), 4)
                                            for punto, color in [(cadera, (0, 255, 0)), (rodilla, (0, 0, 255)), (tobillo, (255, 0, 0))]:
                                                cv2.circle(fotograma, punto, 8, color, -1)
                                            posicion_texto = (max(5, min(rodilla[0] + 12, ancho - 140)), max(25, rodilla[1] - 18))
                                            cv2.putText(fotograma, f"{angulo:.1f} deg", posicion_texto,
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                                escritor.write(fotograma)
                        captura.release()
                        escritor.release()

                        if contador == 0:
                            st.error("El video no contiene fotogramas legibles.")
                        else:
                            st.subheader("Video con análisis de movimiento")
                            procesado_ok, procesado_error = convertir_a_web(ruta_temporal, ruta_procesada)
                            if procesado_ok:
                                with open(ruta_procesada, "rb") as archivo:
                                    video_procesado = archivo.read()
                                st.video(video_procesado, format="video/mp4")
                                st.download_button("Descargar video analizado", video_procesado,
                                                   file_name="sentadilla_analizada.mp4", mime="video/mp4")
                            else:
                                st.error("No fue posible convertir el video analizado para reproducirlo.")
                                st.code(procesado_error or "Error desconocido")

                            if angulos:
                                indice_minimo = int(np.argmin(angulos))
                                minimo = angulos[indice_minimo]
                                rom = max(angulos) - minimo
                                st.subheader("Resultados del análisis")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Ángulo mínimo", f"{minimo:.1f}°")
                                c2.metric("ROM observado", f"{rom:.1f}°")
                                c3.metric("Mayor flexión en", f"{tiempos[indice_minimo]:.2f} s")
                                st.caption("El ángulo mínimo corresponde a la mayor flexión observada. El ROM mostrado es la diferencia entre los ángulos máximo y mínimo detectados.")
                                tabla = pd.DataFrame({"Tiempo (s)": tiempos, "Ángulo de rodilla (°)": angulos})
                                st.subheader("Ángulo de rodilla a lo largo del tiempo")
                                st.line_chart(tabla.set_index("Tiempo (s)"))
                                st.subheader("Datos del análisis")
                                st.dataframe(tabla.iloc[::max(1, len(tabla) // 20)].round(2), hide_index=True, use_container_width=True)
                            else:
                                st.warning("Se procesó el video, pero no se detectaron suficientes puntos de cadera, rodilla y tobillo. Prueba con otra vista lateral o cambia la pierna seleccionada.")

    st.selectbox("Pierna que se analizará", ["Derecha", "Izquierda"], key="lado_rodilla",
                 help="Selecciona la pierna que permanece más visible durante la sentadilla.")

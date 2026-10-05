# ---------------------------------------------
# GRÁFICO
# ---------------------------------------------

st.subheader(
    "Variación del ángulo de rodilla durante la ejecución"
)

# Crear eje temporal original
tiempos = np.arange(len(angulos_rodilla)) / fps

# Crear estructura con todos los datos originales
datos_grafico = {
    "Tiempo (s)": tiempos,
    "Ángulo de rodilla (°)": angulos_rodilla,
}

# Crear gráfico con todos los fotogramas
st.line_chart(
    datos_grafico,
    x="Tiempo (s)",
    y="Ángulo de rodilla (°)",
)

# ---------------------------------------------
# TABLA RESUMIDA POR SEGUNDOS ENTEROS
# ---------------------------------------------

st.subheader("Datos del análisis por segundo")

# Obtener los segundos enteros presentes en el video
segundos = np.arange(
    0,
    int(np.ceil(duracion)) + 1
)

# Evitar mostrar segundos fuera de la duración real
segundos = segundos[segundos <= duracion]

# Buscar el ángulo correspondiente al fotograma
# más cercano a cada segundo
angulos_por_segundo = []

for segundo in segundos:

    indice = int(round(segundo * fps))

    # Evitar superar el último dato disponible
    if indice >= len(angulos_rodilla):
        indice = len(angulos_rodilla) - 1

    angulos_por_segundo.append(
        angulos_rodilla[indice]
    )

# Crear tabla
datos_tabla = {
    "Tiempo (s)": segundos.astype(int),
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

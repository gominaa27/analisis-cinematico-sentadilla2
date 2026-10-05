# 🦿 Análisis Cinemático de la Sentadilla Bipodal (Kinesiología UCh)

Interfaz web interactiva desarrollada para la cuantificación y análisis cinemático del gesto de la sentadilla mediante visión artificial y procesamiento de video digital, en el marco de la asignatura de Bioinstrumental del Movimiento Humano de la Carrera de Kinesiología, Universidad de Chile.

---

## 🚀 Enlace a la Aplicación en Vivo
Puedes probar la aplicación web funcionando directamente en el siguiente enlace de Streamlit Cloud:
👉 **[Acceder a la Aplicación](https://app-sentadilla-uchile.streamlit.app)** *(reemplaza con tu link exacto si difiere)*

---

## 🎯 Objetivo del Proyecto
Superar las limitaciones de la goniometría clínica tradicional (subjetividad y dependencia del observador) mediante el uso de algoritmos de estimación de postura corporal (*MediaPipe Pose*), permitiendo extraer de forma automatizada y objetiva:
* Ángulos instantáneos de flexión de rodilla.
* Rango de movimiento (ROM).
* Tiempos de ejecución y fases del movimiento (excéntrica y concéntrica).

---

## 🛠️ Tecnologías y Librerías Utilizadas
* **Python** como lenguaje base de programación.
* **Streamlit** para el diseño e interactividad de la interfaz de usuario web.
* **MediaPipe (Google)** para la detección de puntos clave corporales (*Pose Estimation*).
* **OpenCV / NumPy / Pandas** para la manipulación de fotogramas de video, cálculos vectoriales y manejo de métricas cuantitativas.

---

## 📂 Estructura del Repositorio
```text
analisis-cinematico-sentadilla2/
│
├── app.py                 # Código principal de la aplicación en Streamlit
├── requirements.txt       # Librerías y dependencias necesarias del proyecto
└── README.md              # Documentación y descripción general del repositorio

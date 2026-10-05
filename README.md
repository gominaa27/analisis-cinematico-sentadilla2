# Análisis Cinemático de Sentadilla

Aplicación web desarrollada en **Streamlit** para el análisis cinemático de la rodilla durante la ejecución de una **sentadilla bipodal libre**, utilizando un registro de video.

La aplicación utiliza **MediaPipe Pose** para estimar la posición de la cadera, rodilla y tobillo del miembro inferior analizado y calcular el ángulo de la rodilla durante la ejecución del gesto.

## Objetivo

Cuantificar el movimiento de la rodilla durante una sentadilla mediante análisis de video, obteniendo indicadores cinemáticos que complementen la observación visual del gesto.

## Funcionamiento

El análisis se realiza mediante los siguientes pasos:

1. El usuario carga un video de la sentadilla.
2. MediaPipe Pose detecta los puntos anatómicos de la cadera, rodilla y tobillo del miembro inferior analizado.
3. Se obtienen sus coordenadas 2D en cada fotograma.
4. Se calcula el ángulo de la rodilla utilizando la rodilla como vértice y el producto punto entre los vectores cadera–rodilla y tobillo–rodilla.
5. Los valores obtenidos se almacenan como una serie temporal.
6. Se calculan los siguientes indicadores:

   * **Máxima flexión de rodilla:** menor ángulo registrado.
   * **ROM de rodilla:** diferencia entre el mayor y menor ángulo registrado.
   * **Tiempo hasta máxima flexión:** instante en que se alcanza el menor ángulo.
   * **Duración del video.**
7. Se genera un video procesado mostrando los puntos anatómicos, los vectores utilizados y el ángulo de la rodilla.
8. Los resultados se presentan mediante métricas, un gráfico de ángulo versus tiempo y una tabla con valores por segundo.

## Variable analizada

La variable principal es el **ángulo de la articulación de la rodilla**, expresado en grados (°).

El cálculo utiliza las coordenadas 2D de:

* Cadera
* Rodilla
* Tobillo

La rodilla corresponde al vértice del ángulo.

## Requisitos del video

Para obtener mejores resultados se recomienda:

* Vista lateral o cercana al plano sagital.
* Persona visible de cuerpo completo.
* Cadera, rodilla y tobillo visibles durante toda la ejecución.
* Buena iluminación.
* Cámara estable.
* Evitar obstáculos u oclusiones importantes.

### Formatos aceptados

* `.mp4`
* `.mov`
* `.avi`
* `.webm`

## Uso de la aplicación

1. Ingresar a la aplicación web.
2. Seleccionar **“Sube tu video aquí”**.
3. Cargar el archivo de video.
4. Seleccionar **“Analizar gesto”**.
5. Esperar el procesamiento.
6. Revisar el video procesado y los indicadores obtenidos.
7. Analizar el gráfico de la variación del ángulo de rodilla y la tabla de valores por segundo.

## Aplicación web

**Aplicación:**
https://app-sentadilla-uchile.streamlit.app/

**Repositorio:**
https://github.com/gominaa27/analisis-cinematico-sentadilla2

## Tecnologías utilizadas

* **Python 3.11**
* **Streamlit**
* **MediaPipe Pose**
* **OpenCV**
* **NumPy**
* **FFmpeg**

## Instalación local

Clonar el repositorio:

```bash
git clone https://github.com/gominaa27/analisis-cinematico-sentadilla2.git
cd analisis-cinematico-sentadilla2
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

La aplicación se abrirá localmente en el navegador.

## Dependencias

El archivo `requirements.txt` contiene:

```text
streamlit
opencv-python-headless==4.10.0.84
numpy
mediapipe==0.10.14
```

El archivo `runtime.txt` especifica la versión de Python:

```text
python-3.11
```

El archivo `packages.txt` contiene:

```text
ffmpeg
```

FFmpeg se utiliza para convertir el video procesado a un formato compatible con su reproducción en navegadores web.

## Estructura del proyecto

```text
analisis-cinematico-sentadilla2/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── packages.txt
└── README.md
```

## Reproducibilidad

El proyecto incluye el código fuente, las dependencias y la configuración necesaria para ejecutar la aplicación. Para reproducir el análisis localmente se requiere **Python 3.11**, instalar las dependencias indicadas y ejecutar `app.py` mediante Streamlit.

## Consideraciones y limitaciones

El análisis corresponde a una estimación bidimensional basada en video. La precisión de la estimación depende de la calidad del registro, la iluminación, la posición de la cámara y la visibilidad de los segmentos corporales.

Los resultados tienen fines académicos y de análisis del movimiento, por lo que no reemplazan una evaluación clínica o biomecánica instrumental.

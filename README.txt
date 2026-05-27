# DriveTest Studio

DriveTest Studio es una aplicación desarrollada en Python para el análisis de campañas de medida de redes móviles 3G, 4G y 5G.

El proyecto ha sido desarrollado como parte de un Trabajo Fin de Máster en Ingeniería de Telecomunicación. La herramienta permite cargar automáticamente archivos de campaña, procesar registros de red, fusionar medidas con resultados de Speedtest, representar mapas de cobertura, detectar handovers, comparar tecnologías y generar predicciones de cobertura mediante aprendizaje automático.

## Funcionalidades principales

- Detección automática de campañas de medida.
- Lectura de archivos GMON, CLF, KML y SPEEDTEST.
- Representación de rutas sobre mapas.
- Detección y visualización de eventos de handover.
- Generación de heatmaps de potencia recibida.
- Comparación entre campañas 3G, 4G y 5G.
- Estadísticas de calidad de señal, rendimiento y jitter.
- Predicción de cobertura mediante Random Forest.
- Incorporación de edificios del entorno mediante OpenStreetMap/Catastro.

## Tecnologías utilizadas

- Python
- Tkinter
- pandas
- NumPy
- Matplotlib
- scikit-learn
- OpenStreetMap
- Contextily

## Campaña de medidas

La validación del programa se realizó mediante una campaña de medidas en Puerto de Sagunto con el operador O2. Se realizaron recorridos físicos para escenarios 3G, 4G y 5G, registrando parámetros radio y pruebas de velocidad periódicas.

En 3G se utiliza RSCP como métrica de potencia recibida, mientras que en 4G y 5G se emplea RSRP o SS-RSRP. Para facilitar la comparación conjunta, el programa representa estas magnitudes como potencia recibida en dBm.

## Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt
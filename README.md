# Ejecución del Script `tarea1.py`

Este proyecto utiliza un entorno virtual para gestionar las dependencias necesarias para ejecutar el script `tarea1.py`.

## Pasos para la ejecución

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python tarea1.py



Al ser ejecutado, se mostrará una matriz de confusion del Random Forest, 
se debe cerrar para mostrar la siguiente que será la de Regresión logística y luego al cerrarla, mostrará la de KNN. 
Además en terminar se mostrarán las matrices como tablas. 
Finalmente por la terminal se imprime una tabla comparando los modelos con las metricas.
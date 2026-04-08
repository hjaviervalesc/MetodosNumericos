#Estructura del proyecto
El proyecto está dividido en 6 scripts. Cada uno de ellos hace referencia a una etapa del enunciado del proyecto "proyecto_final_implementacion_guiada".
    - etapa0.py añade fuentes de densidad a la simulación y prepara todo lo referente al renderizado con taichi lang.
    - etapa1.py difusión de la densidad utilizando el método iteractivo Jacobi.
    - etapa2.py advección de la densidad por el campo de velocidad.
    - etapa3.py añade fuerzas externas para generar movimiento.
    - etapa4.py difusión y advección de la velocidad.
    - etapa5.py proyección para incomprensibilidad.
Cada uno de estos scripts tiene únicamente descrito el código que ataja a esa étapa, por lo que para encontrar lo referente a las otras etapas hay que ir directamente a esas etapas.

#Ejecución de los scripts
Los scripts están acondicionados para que al accionar el play aparezca la simulación directamente, por lo que no hay que hacer nada más que dar al play, en cada script.

#Interacción del usuario con la simulación
Del script etapa0.py al script etapa2.py se añade fuentes de densidad pulsando el click derecho del ratón.
Del script etapa3.py al script etapa5.py aparte de las fuentes de densidad, se añaden fuerzas externas pulsando el click izquierdo.

#Referencias
https://docs.taichi-lang.org/
https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/ggui_examples/stable_fluid_ggui.py
Real-Time Fluid Dynamics for Games - Paper - Jos Stam
Real-Time Fluid Dynamics for Games - Slides - Jos Stam
Profesor de la asignatura Diego Rojo
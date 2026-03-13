# Referencia: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/ggui_examples/stable_fluid_ggui.py

import numpy as np
import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)


class FieldPair:
    def __init__(self, current_field, next_field):
        self.cur = current_field
        self.nxt = next_field

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


# Parametros de la simulacion de fluidos
res = 512  # Resolución del grid
h = 1 / res  # Tamaño de la celda
dt = 0.03  # Tamaño del paso de tiempo

## Parametros de la fuente de densidad
s_dens = 10
s_radius = res / 15.0

# Estructuras de datos
## Campos de densidad
_density_field_1 = ti.field(float, shape=(res,res)) #Grid de initial state
_density_field_2 = ti.field(float, shape=(res,res)) #Grid de add sources


dens = FieldPair(_density_field_1, _density_field_2)

#Cabecera con la que taichi toma el control de las tareas paralelas
@ti.kernel
                #pilla los grids     #Eventos de ratón
def add_sources(dens: ti.template(), input_data: ti.types.ndarray()):
    #Para cada casilla de mis grids
    for i, j in dens:
        densidad = input_data[2] * s_dens #Actualiza la cantidad de densidad a añadir
                #eje de mi ratón
        mx, my = input_data[0], input_data[1]
        #Las celdas de taichi default son 1x1 
        # Centro casilla i,j
        cx = i + 0.5
        cy = j + 0.5
        # Distancia al centro de la casilla (al cuadrado), de mi ratón
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        # Decaimiento exponencial
        dens[i, j] += dt * densidad * ti.exp(-6 * d2 / s_radius**2) #+ tiempo más cantidad y difumina con las celdas cercanas según su distancia


def init():
    #Hago 0s en las matrices
    dens.cur.fill(0) 
    dens.nxt.fill(0)


def step(input_data):
    #Se va a ir actualizando en cada frame
    add_sources(dens.cur, input_data)


def main():
    paused = False
    #Creo la ventana
    window = ti.ui.Window("Stable Fluids", (res, res), vsync=True)
    canvas = window.get_canvas()

    # Inicialización
    init()

    # Bucle Principal
    #Actualización por frame
    while window.running:
        # 0: mouse_x 1: mouse_y 2: source_active
        input_data = np.zeros(3, dtype=np.float32)

        # Input Teclado
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:  # ESC para salir
                break
            elif e.key == "r":  # 'r' para resetear
                paused = False
                init()
            elif e.key == "p":  # 'p' para pausar/reanudar
                paused = not paused

        # Input Ratón
        if window.is_pressed(ti.ui.RMB):
            #Monitoreo la posición de mi ratón
            mouse_xy = window.get_cursor_pos()
            input_data[0] = mouse_xy[0] * res
            input_data[1] = mouse_xy[1] * res
            input_data[2] = 1.0

        # Simulación (siguiente paso de tiempo)
        if not paused:
            step(input_data)

        # Renderizado
        # Donde se va a mostrar el resultado
        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()

# Referencia: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/ggui_examples/stable_fluid_ggui.py

import numpy as np
import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch, debug=True)
print("Backend activo:", ti.cfg.arch) 
#Backedn activo: Arch.vulkan, vulkan trabaja en GPU


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
#Grids
_density_field_1 = ti.field(float, shape=(res,res)) 
_density_field_2 = ti.field(float, shape=(res,res)) 
dens = FieldPair(_density_field_1, _density_field_2)

## Campos de velocidad
#Grids
#Velocidad horizontal
_vel_u_1 = ti.field(float, shape=(res, res))
_vel_u_2 = ti.field(float, shape=(res, res))
vel_u = FieldPair(_vel_u_1, _vel_u_2)
#Velocidad vertical
_vel_v_1 = ti.field(float, shape=(res, res))
_vel_v_2 = ti.field(float, shape=(res, res))
vel_v = FieldPair(_vel_v_1, _vel_v_2)                


#Cabecera con la que taichi toma el control de las tareas 
@ti.kernel
                #pilla los grids     #Eventos de ratón
def add_sources(dens: ti.template(), input_data: ti.types.ndarray()):
    #Para cada casilla de mis grids
    for i, j in dens.cur:
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
        dens.cur[i, j] += dt * densidad * ti.exp(-6 * d2 / s_radius**2) #+ tiempo más cantidad según la distancia del raton con la celda

#Cabecera con la que taichi toma el control de las tareas 
@ti.kernel
def add_forces(vel_u: ti.template(), vel_v: ti.template(), input_data: ti.types.ndarray()):
    #Para cada casilla de mis grids
    for i, j in vel_u.cur:
        mx, my = input_data[3], input_data[4]
        #Las celdas de taichi default son 1x1 
        # Centro casilla i,j
        cx = i + 0.5
        cy = j + 0.5
        # Distancia al centro de la casilla (al cuadrado), de mi ratón
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        vel_u.cur[i,j] = mx * ti.exp(-2 * d2 / s_radius**2) * 10 * input_data[5]
        vel_v.cur[i,j] = my * ti.exp(-2 * d2 / s_radius**2) * 10 * input_data[5]

#Una vez por cada iteración de jacobi
@ti.kernel
def jacobi_iter(dens_cur: ti.template(), dens_nxt: ti.template(), a:float):
    #Paralelizando en GPU
    ti.loop_config(block_dim=16)
    for i, j in dens_cur:
        #Excluyo las casillas del marco de mi grid
        if i == 0 or j == 0 or i == res-1 or j == res-1:
            continue
        #sin + porque trabajo con el valor de la iteracion anterior para todas las celdas, sino es Gauss-Seidel no Jacobi
        dens_nxt[i,j] = (dens_cur[i,j] + a*(dens_cur[i-1,j]+dens_cur[i+1,j]+dens_cur[i,j-1]+dens_cur[i,j+1]))/(1+4*a)

@ti.kernel
def set_boundaries(dens: ti.template()):
    for i, j in dens:
        #La fila de arriba excepto la casilla primera y final
        if i == 0 and 0 < j < res - 1:
            dens[i,j] = dens[i+1,j]
        #La fila de abajo excepto la casilla primera y final
        elif i == res - 1 and 0 < j < res - 1:
            dens[i, j] = dens[i - 1, j]
        #La fila del lateral izq excepto la casilla primera y final
        elif j == 0 and 0 < i < res - 1:
            dens[i, j] = dens[i, j + 1]
        #La fila del lateral derecho excepto la casilla primera y final
        elif j == res - 1 and 0 < i < res - 1:
            dens[i, j] = dens[i, j - 1]
            
    #Las casillas de las esquina
    dens[0,0] = (dens[0,1] + dens[1,0]) / 2 #Arriba izq
    dens[res-1,0] = (dens[res-1,1] + dens[res-2,0]) / 2 #Abajo izquierda
    dens[0, res-1] =(dens[0,res-2] + dens[1,res-1]) / 2 #Arriba derecha
    dens[res-1, res-1] = (dens[res-1,res-2] + dens[res-2,res-1]) / 2 #Abajo derecha
          
def diffuse(dens: ti.template(), dt:float, h:float, k:float):
    a = k * dt / h**2 #Solo de una celda
    #Numero de iteraciones en bucle
    for it in range(30): 
        jacobi_iter(dens.cur,dens.nxt,a)
        #Buffer auxiliar
        dens.swap()
        set_boundaries(dens.cur)


#Para funciones que se ejecutan dentro de kernels
@ti.func
def bilerp(dens_cur:ti.template(), x, y):
    #Veo posiciones en las esquinas de la casilla que engloba al punto x,y
    #Trunco
    i0 = int(ti.floor(x))
    j0 = int(ti.floor(y))
    i1 = i0 + 1
    j1 = j0 + 1

    #Pesos en la interpolación según lo lejos que tengo las esquinas
    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1

    #Interpolo
    return s0*(t0*dens_cur[i0,j0] + t1* dens_cur[i0, j1]) + s1*(t0*dens_cur[i1, j0] + t1*dens_cur[i1, j1])

@ti.kernel
def advect(dens:ti.template(), dt:float, vel_u:ti.template(), vel_v:ti.template()):
    for i, j in dens.nxt:
        #Vectores velocidad
        u = vel_u.cur[i, j]
        v = vel_v.cur[i, j]
        #Centros
        cx = i + 0.5
        cy = j + 0.5
        #Posición backwards en el campo velocidad
        x = cx - dt * u
        y = cy - dt * v
        #Controlo no salir del grid
        x = ti.math.clamp(x, 0.5, res - 1.5)
        y = ti.math.clamp(y, 0.5, res - 1.5)
        #Interpolo
        dens.nxt[i,j] = bilerp(dens.cur, x, y)
    set_boundaries(dens.nxt)

@ti.kernel
def vel_horizontal(vel_u_cur: ti.template()):
    for i, j in vel_u_cur:
        if 0 <= j < 150:
            vel_u.cur[i, j] = j * -0.01
        elif 150 <= j <= 300:
            vel_u.cur[i, j] =  j * 0.15
        elif j > 300:
            vel_u.cur[i, j] = j * -0.01


def init():
    #Hago 0s en las matrices
    dens.cur.fill(0) 
    dens.nxt.fill(0)
    vel_u.cur.fill(0) 
    vel_u.nxt.fill(0)
    vel_v.cur.fill(0) 
    vel_v.nxt.fill(0)


def step(input_data):
    #Se va a ir actualizando en cada frame
    add_sources(dens, input_data)
    add_forces(vel_u, vel_v, input_data)
    set_boundaries(dens.cur)
    diffuse(dens, dt,h, k = 3.0e-5)
    dens.swap()
    advect(dens,dt,vel_u,vel_v)


def main():
    paused = False
    forces = True
    #Creo la ventana
    window = ti.ui.Window("Stable Fluids", (res, res), vsync=True)
    canvas = window.get_canvas()
    # Inicialización
    init()

    # Bucle Principal
    #Actualización por frame
    while window.running:
        # 0: mouse_x 1: mouse_y 2: source_active
        input_data = np.zeros(6, dtype=np.float32) # Esto es S.

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

        # Input Ratón Derecho
        if window.is_pressed(ti.ui.RMB):
            #Monitoreo la posición de mi ratón
            mouse_xy = window.get_cursor_pos()
            input_data[0] = mouse_xy[0] * res
            input_data[1] = mouse_xy[1] * res
            input_data[2] = 1.0
        if window.is_pressed(ti.ui.LMB):
            #Monitoreo la posición de mi ratón
            mouse_xy = window.get_cursor_pos()
            input_data[3] = mouse_xy[0] * res
            input_data[4] = mouse_xy[1] * res
            input_data[5] = 1.0
            
        # Simulación (siguiente paso de tiempo)
        if not paused:
            step(input_data)

        # Renderizado
        # Donde se va a mostrar el resultado
        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()

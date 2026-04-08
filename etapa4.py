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


##Parametros de fuente de fuerza
s_fuerza = 100

# Estructuras de datos
## Campos de densidad
#Grid
_density_field_1 = ti.field(float, shape=(res,res)) 
_density_field_2 = ti.field(float, shape=(res,res)) 
dens = FieldPair(_density_field_1, _density_field_2)
## Campos de velocidad
#Grid
#Velocidad horizontal
_vel_u_1 = ti.field(float, shape=(res, res))
_vel_u_2 = ti.field(float, shape=(res, res))
vel_u = FieldPair(_vel_u_1, _vel_u_2)
#Velocidad vertical
_vel_v_1 = ti.field(float, shape=(res, res))
_vel_v_2 = ti.field(float, shape=(res, res))
vel_v = FieldPair(_vel_v_1, _vel_v_2)  


@ti.kernel
def add_sources(dens: ti.template(), input_data: ti.types.ndarray()):
    for i, j in dens.cur:
        densidad = input_data[2] * s_dens 
        mx, my = input_data[0], input_data[1]
        cx = i + 0.5
        cy = j + 0.5
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        dens.cur[i, j] += dt * densidad * ti.exp(-6 * d2 / s_radius**2)

@ti.kernel
def add_forces(vel_u: ti.template(), vel_v: ti.template(), input_data: ti.types.ndarray()):
    for i, j in vel_u.cur:
        fuerza = input_data[5] * s_fuerza 
        
        mx, my = input_data[3], input_data[4]
        dx, dy = input_data[6], input_data[7]

        cx = i + 0.5
        cy = j + 0.5

        d2 = (cx - mx) ** 2 + (cy - my) ** 2

        vel_u.cur[i,j] += dx * ti.exp(-6 * d2 / s_radius**2) * fuerza
        vel_v.cur[i,j] += dy * ti.exp(-6 * d2 / s_radius**2) * fuerza

#Adaptado tanto para campos de velocidad como de densidad
@ti.kernel
def jacobi_iter(field_cur: ti.template(), field_nxt: ti.template(), a:float):
    for i, j in field_cur:
        if 0 < i < res - 1 and 0 < j < res - 1:
            field_nxt[i,j] = (field_cur[i,j] + a*(field_cur[i-1,j]+field_cur[i+1,j]+field_cur[i,j-1]+field_cur[i,j+1]))/(1+4*a)


@ti.kernel
def set_boundaries(dens: ti.template()):
    for i, j in dens:
        if i == 0 and 0 < j < res - 1:
            dens[i,j] = dens[i+1,j]
        elif i == res - 1 and 0 < j < res - 1:
            dens[i, j] = dens[i - 1, j]
        elif j == 0 and 0 < i < res - 1:
            dens[i, j] = dens[i, j + 1]
        elif j == res - 1 and 0 < i < res - 1:
            dens[i, j] = dens[i, j - 1]
            
    dens[0,0] = (dens[0,1] + dens[1,0]) / 2 #Arriba izq
    dens[res-1,0] = (dens[res-1,1] + dens[res-2,0]) / 2 #Abajo izquierda
    dens[0, res-1] =(dens[0,res-2] + dens[1,res-1]) / 2 #Arriba derecha
    dens[res-1, res-1] = (dens[res-1,res-2] + dens[res-2,res-1]) / 2 #Abajo derecha


@ti.kernel
def set_boundaries_reflective(vel_u_cur: ti.template(), vel_v_cur: ti.template()):
    for i, j in vel_u_cur:
        # La fila de arriba excepto la casilla primera y final
        if i == 0 and 0 < j < res - 1:
            vel_u_cur[i,j] =  vel_u_cur[i+1,j]    # tangencial copiada
            vel_v_cur[i,j] = -vel_v_cur[i+1,j]    # normal invertida

        # La fila de abajo excepto la casilla primera y final
        elif i == res - 1 and 0 < j < res - 1:
            vel_u_cur[i, j] =  vel_u_cur[i - 1, j]   # tangencial copiada
            vel_v_cur[i, j] = -vel_v_cur[i - 1, j]   # normal invertida  

        # La fila del lateral izq excepto la casilla primera y final
        elif j == 0 and 0 < i < res - 1:
            vel_u_cur[i, j] = -vel_u_cur[i, j + 1]   # normal invertida
            vel_v_cur[i, j] =  vel_v_cur[i, j + 1]   # tangencial copiada    

        # La fila del lateral derecho excepto la casilla primera y final
        elif j == res - 1 and 0 < i < res - 1:
            vel_u_cur[i, j] = -vel_u_cur[i, j - 1]   # normal invertida
            vel_v_cur[i, j] =  vel_v_cur[i, j - 1]   # tangencial copiada
            
    #Las casillas de las esquina
    vel_u_cur[0,0] = (vel_u_cur[0,1] + vel_u_cur[1,0]) / 2 #Arriba izq
    vel_v_cur[0,0] = (vel_v_cur[0,1] + vel_v_cur[1,0]) / 2 #Arriba izq

    vel_u_cur[res-1,0] = (vel_u_cur[res-1,1] + vel_u_cur[res-2,0]) / 2 #Abajo izquierda
    vel_v_cur[res-1,0] = (vel_v_cur[res-1,1] + vel_v_cur[res-2,0]) / 2 #Abajo izquierda

    vel_u_cur[0, res-1] =(vel_u_cur[0,res-2] + vel_u_cur[1,res-1]) / 2 #Arriba derecha
    vel_v_cur[0, res-1] =(vel_v_cur[0,res-2] + vel_v_cur[1,res-1]) / 2 #Arriba derecha

    vel_u_cur[res-1, res-1] = (vel_u_cur[res-1,res-2] + vel_u_cur[res-2,res-1]) / 2 #Abajo derecha
    vel_v_cur[res-1, res-1] = (vel_v_cur[res-1,res-2] + vel_v_cur[res-2,res-1]) / 2 #Abajo derecha

#Divido diffuse en dos funciones, una para densidad y otra para velocidad, porque la función de set_boundaries es diferente para cada una de ellas.
def diffuse_dens(dens: ti.template(), dt:float, h:float, k:float):
    a = k * dt / h**2 
    for it in range(30): 
        jacobi_iter(dens.cur,dens.nxt,a)
        dens.swap()
        set_boundaries(dens.cur)

def diffuse_vel(vel_u: ti.template(),vel_v: ti.template(), dt:float, h:float, visc:float):
    a = visc * dt / h**2 
    for it in range(30): 
        jacobi_iter(vel_u.cur,vel_u.nxt,a)
        vel_u.swap()
        jacobi_iter(vel_v.cur,vel_v.nxt,a)
        vel_v.swap()
        set_boundaries_reflective(vel_u.cur, vel_v.cur)
 
#Adecuado tanto para campos de velocidad como de densidad
@ti.func
def bilerp(field_cur:ti.template(), x, y):
    i0 = int(ti.floor(x))
    j0 = int(ti.floor(y))

    i1 = i0 + 1
    j1 = j0 + 1

    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1

    return s0*(t0*field_cur[i0,j0] + t1* field_cur[i0, j1]) + s1*(t0*field_cur[i1, j0] + t1*field_cur[i1, j1])

#Adecuado tanto para campos de velocidad como de densidad
@ti.kernel
def advect(field:ti.template(), dt:float, vel_u:ti.template(), vel_v:ti.template()):
    for i, j in field.nxt:
        u = vel_u.cur[i, j]
        v = vel_v.cur[i, j]

        cx = i + 0.5
        cy = j + 0.5

        x = cx - dt * u
        y = cy - dt * v

        x = ti.math.clamp(x, 0.5, res - 1.5)
        y = ti.math.clamp(y, 0.5, res - 1.5)

        field.nxt[i,j] = bilerp(field.cur, x, y)

#Divido advect en dos funciones, una para densidad y otra para velocidad, porque la función de set_boundaries es diferente para cada una de ellas.
def advect_vel(vel_u:ti.template(), vel_v:ti.template(), dt:float):
    advect(vel_u, dt, vel_u, vel_v)
    advect(vel_v, dt, vel_u, vel_v)
    vel_u.swap()
    vel_v.swap()
    set_boundaries_reflective(vel_u.cur, vel_v.cur)

def advect_dens(dens:ti.template(), dt:float, vel_u:ti.template(), vel_v:ti.template()):
    advect(dens, dt, vel_u, vel_v)
    dens.swap()
    set_boundaries(dens.cur)    

@ti.kernel
def vel_horizontal(vel_u_cur: ti.template()):
    for i, j in vel_u_cur:
        if 0 <= j < 150:
            vel_u.cur[i, j] =  -10
        elif 150 <= j <= 300:
            vel_u.cur[i, j] =  30
        elif j > 300:
            vel_u.cur[i, j] =  -10


def init():
    #Hago 0s en las matrices
    dens.cur.fill(0) 
    dens.nxt.fill(0)
    vel_u.cur.fill(0) 
    vel_u.nxt.fill(0)
    vel_v.cur.fill(0) 
    vel_v.nxt.fill(0)


def dens_step(input_data):
    add_sources(dens, input_data)
    set_boundaries(dens.cur)
    diffuse_dens(dens, dt,h, k = 3.0e-5)
    advect_dens(dens,dt,vel_u,vel_v)

def vel_step(input_data):
    add_forces(vel_u, vel_v, input_data)
    set_boundaries_reflective(vel_u.cur, vel_v.cur)
    diffuse_vel(vel_u,vel_v, dt,h, visc = 3.0e-5)
    advect_vel(vel_u, vel_v, dt)


def step(input_data):
    dens_step(input_data)
    vel_step(input_data)


def main():
    paused = False
    #Creo la ventana
    window = ti.ui.Window("Stable Fluids", (res, res), vsync=True)
    canvas = window.get_canvas()
    # Inicialización
    init()
    #frame anterior
    input_data_prev = np.zeros(2, dtype=np.float32) 

    # Bucle Principal
    #Actualización por frame
    while window.running:
        # 0: mouse_x 1: mouse_y 2: source_active
        input_data = np.zeros(8, dtype=np.float32) # Esto es S.

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

        # Input Ratón derecho
        if window.is_pressed(ti.ui.RMB):
            #Monitoreo la posición de mi ratón
            mouse_xy = window.get_cursor_pos()
            input_data[0] = mouse_xy[0] * res
            input_data[1] = mouse_xy[1] * res
            input_data[2] = 1.0
            
        # Input Ratón izquierdo
        if window.is_pressed(ti.ui.LMB):
            #Monitoreo la posición de mi ratón
            mouse_xy = window.get_cursor_pos()
            input_data[3] = mouse_xy[0] * res
            input_data[4] = mouse_xy[1] * res
            input_data[5] = 1.0 
            input_data[6] = input_data[3] - input_data_prev[0]
            input_data[7] = input_data[4] - input_data_prev[1]
            input_data_prev[0] = input_data[3]
            input_data_prev[1] = input_data[4]

        # Simulación (siguiente paso de tiempo)
        if not paused:
            step(input_data)

        # Renderizado
        # Donde se va a mostrar el resultado
        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()

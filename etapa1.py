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
#Grid
_density_field_1 = ti.field(float, shape=(res,res)) 
_density_field_2 = ti.field(float, shape=(res,res)) 

dens = FieldPair(_density_field_1, _density_field_2)

@ti.kernel
def add_sources(dens: ti.template(), input_data: ti.types.ndarray()):
    for i, j in dens.cur:
        densidad = input_data[2] * s_dens 
        mx, my = input_data[0], input_data[1]
        cx = i + 0.5
        cy = j + 0.5
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        dens.cur[i, j] += dt * densidad * ti.exp(-6 * d2 / s_radius**2)

#Una vez por cada iteración, kernel ya paraleliza
@ti.kernel
def jacobi_iter(dens_cur: ti.template(), dens_nxt: ti.template(), a:float):
    for i, j in dens_cur:
        #Evita acceso fuera de rangos del grid
        if 0 < i < res - 1 and 0 < j < res - 1:
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
    #Ratio de difusión, cuanto más grande más se va a difundir la cantidad de densidad.
    a = k * dt / h**2 
    #Numero de iteraciones en bucle
    for it in range(150): 
        jacobi_iter(dens.cur,dens.nxt,a)
        #Buffer auxiliar
        dens.swap() #Necesario porque sino no se está actualizando los grid se está haciendo lo mismo una y otra vez 
        set_boundaries(dens.cur)

def init():
    dens.cur.fill(0) 
    dens.nxt.fill(0)

def step(input_data):
    #Se va a ir actualizando en cada frame
    add_sources(dens, input_data)
    diffuse(dens, dt,h, k = 3.0e-5)

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
        input_data = np.zeros(3, dtype=np.float32) # Esto es S.

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

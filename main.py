#Simulación de Fluidos en Tiempo Real para Videojuegos
import numpy as np
import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)
#ti.init(arch=arch.cpu, debug=True) #Interesante. Para saber en que linea del kernel estoy accediendo y no puedo

class FieldPair:
    def __init__(self, current_field, next_field):
        self.cur = current_field
        self.nxt = next_field

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

# Parámetros de la simulacion de fluidos
res = 512  # Resolución del grid
h = 1 / res  # Tamaño de la celda
dt = 0.03  # Tamaño del paso de tiempo

## Parametros de la fuente de densidad
s_dens = 10.0 
s_radius = res / 15.0

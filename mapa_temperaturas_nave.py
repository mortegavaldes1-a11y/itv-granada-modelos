import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, Circle

# =========================
# 🔧 PARÁMETROS CONTROLABLES
# =========================

# Velocidades (m/s)
V_entrada = 1.5       # aire que entra por la puerta de entrada
V_salida = 0.8        # aire que sale por la puerta de salida
V_conos = 0.6         # velocidad de impulsión vertical

# Apertura de puertas (0 = cerrada, 1 = abierta)
apertura_entrada = 1.0
apertura_salida = 1.0

# Temperaturas (ºC)
T_ext = 45
T_impulsion = 26
T_vehiculo = 80

# =========================
# 📐 GEOMETRÍA
# =========================

Lx = 42
Ly = 6.5

dx = 0.5
dy = 0.5

nx = int(Lx/dx)
ny = int(Ly/dy)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# =========================
# 🌡️ TEMPERATURA
# =========================

T = np.ones((ny, nx)) * 38

k_difusion = 0.15
k_conos = 0.5
k_vehiculos = 0.3

# Conos
conos = [(2,5.8),(3.5,5.8),(5,5.8),
         (13,5.8),(14.5,5.8),(16,5.8),
         (36,5.8)]

# Vehículos
vehiculos = [(4,3),(15,3),(30,3)]

steps = 400

for step in range(steps):
    T_new = T.copy()

    for j in range(1, ny-1):
        for i in range(1, nx-1):

            diffusion = k_difusion * (
                T[j+1,i] + T[j-1,i] + T[j,i+1] + T[j,i-1] - 4*T[j,i]
            )

            # Entrada aire caliente
            entrada = apertura_entrada * V_entrada * 0.1 * (T_ext - T[j,i]) * np.exp(-x[i]/5)

            # Salida aire (reduce temperatura interior)
            salida = -apertura_salida * V_salida * 0.1 * (T[j,i] - T_ext) * np.exp(-(Lx-x[i])/5)

            # Conos
            enfriamiento = 0
            for cx, cy in conos:
                dist = np.sqrt((x[i]-cx)**2 + (y[j]-cy)**2)
                enfriamiento += k_conos * (T_impulsion - T[j,i]) * np.exp(-dist/1.5)

            # Vehículos
            calor = 0
            for vx, vy in vehiculos:
                dist = np.sqrt((x[i]-vx)**2 + (y[j]-vy)**2)
                calor += k_vehiculos * (T_vehiculo - T[j,i]) * np.exp(-dist/1.2)

            T_new[j,i] = T[j,i] + diffusion + entrada + salida + enfriamiento + calor

    T = T_new

# =========================
# 🌪️ CAMPO DE VELOCIDAD
# =========================

U = np.zeros_like(T)
V = np.zeros_like(T)
mask_solido = np.zeros_like(T, dtype=bool)

# Máscara sólidos
for j in range(ny):
    for i in range(nx):
        for vx, vy in vehiculos:
            if abs(x[i]-vx) < 2.25 and abs(y[j]-vy) < 1:
                mask_solido[j,i] = True
        for cx, cy in conos:
            if np.sqrt((x[i]-cx)**2 + (y[j]-cy)**2) < 0.5:
                mask_solido[j,i] = True

# Flujo
for j in range(ny):
    for i in range(nx):

        if mask_solido[j,i]:
            continue

        # Entrada (vector en dirección positiva eje X)
        U[j,i] += apertura_entrada * V_entrada * np.exp(-x[i]/10)

        # Salida (CORREGIDO: vector en dirección positiva eje X para simccionar/extraer el aire)
        U[j,i] += apertura_salida * V_salida * np.exp(-(Lx-x[i])/5)

        # Conos (vertical, hacia abajo)
        for cx, cy in conos:
            dist = np.sqrt((x[i]-cx)**2 + (y[j]-cy)**2)
            if dist < 3:
                V[j,i] -= V_conos * np.exp(-dist/1.5)

        # Recirculación vehículos
        for vx, vy in vehiculos:
            dx_v = x[i] - vx
            dy_v = y[j] - vy
            dist = np.sqrt(dx_v**2 + dy_v**2)
            if dist < 4:
                U[j,i] += -0.5 * dy_v * np.exp(-dist/2)
                V[j,i] += 0.5 * dx_v * np.exp(-dist/2)

# Magnitud velocidad
V_mag = np.sqrt(U**2 + V**2)

# =========================
# 🎨 COLORMAP
# =========================

colors = [
    (0.0, "#0033cc"),
    (0.2, "#66ccff"),
    (0.4, "#ffff66"),
    (0.7, "#ff9933"),
    (1.0, "#cc0000")
]

cmap_custom = LinearSegmentedColormap.from_list("custom", colors)
norm = TwoSlopeNorm(vmin=25, vcenter=35, vmax=60)

# =========================
# 📊 VISUALIZACIÓN
# =========================

plt.figure(figsize=(13,4))
ax = plt.gca()

im = ax.imshow(
    T,
    origin='lower',
    cmap=cmap_custom,
    extent=[0,Lx,0,Ly],
    norm=norm,
    alpha=0.95
)

# Flujo
U_plot = np.ma.masked_where(mask_solido, U)
V_plot = np.ma.masked_where(mask_solido, V)

ax.streamplot(
    X, Y, U_plot, V_plot,
    color='white',
    density=0.8,
    linewidth=1
)

# Vehículos
for vx, vy in vehiculos:
    rect = Rectangle((vx-2.25, vy-1),4.5,2,
                     edgecolor='white',
                     facecolor='black')
    ax.add_patch(rect)

# Conos
for cx, cy in conos:
    circ = Circle((cx, cy),0.5,
                  edgecolor='blue',
                  facecolor='blue')
    ax.add_patch(circ)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label("Temperatura (ºC)")

plt.title("Modelo térmico + flujo configurable")
plt.xlabel("Longitud (m)")
plt.ylabel("Ancho (m)")
plt.axis('equal')

plt.show()

# =============================================================================
# MODELO V8.6 — CORRECCIÓN DE ERROR DE MATRIZ EN CÁLCULO DE VELOCIDAD
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- 1. CONFIGURACIÓN (Panel de control) ---
T_lateral = 40.0        # Temperatura ambiente exterior (ºC)
V_lateral = 1.05        # Viento transversal casi nulo (calma) (m/s)
T_impulsion = 20.0      # Temperatura de las toberas (ºC)
V_impulsion = 6.0       # Velocidad de salida (m/s)
caudal_total_m3h = 1500.0 # Caudal total del sistema (m3/h)
N_toberas = 5           
separacion = 0.75       
altura_tobera = 2.60    

# --- 2. MOTOR FÍSICO DE ALTA RESOLUCIÓN ---
caudal_por_tobera_m3s = (caudal_total_m3h / N_toberas) / 3600.0
A0 = caudal_por_tobera_m3s / V_impulsion
D0 = np.sqrt(4.0 * A0 / np.pi)
M0 = 1.18 * A0 * V_impulsion**2  
centros_toberas = (np.arange(N_toberas) - (N_toberas - 1) / 2.0) * separacion

# Resolución de 1 cm (dx = dy = 0.01)
Lx, Ly = 6.0, 3.0
dx, dy = 0.01, 0.01
nx, ny = int(Lx/dx), int(Ly/dy)
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

T_num = np.zeros_like(X)
T_den = np.zeros_like(X)
V_vel_total = np.zeros_like(X)
Influencia_total = np.zeros_like(X)

for cx in centros_toberas:
    s = np.maximum(altura_tobera - Y, 0.0)
    b = D0/2.0 + 0.08 * s  # Coeficiente de expansión para chorros libres
    desp_x = (1.18 * max(V_lateral, 0.01)**2 * D0 * s**2) / (2.0 * M0 + 1e-6)
    
    # CORREGIDO: Uso de np.maximum en lugar de max() para arrays de NumPy
    denominador_Uc = np.sqrt(2.0) * np.maximum(2 * b, D0)
    Uc = np.clip(V_impulsion * (D0 / denominador_Uc), 0.01, V_impulsion)
    
    # Corrección de entrainment para evitar sobrecalentamiento instantáneo
    Q_local = np.pi * Uc * (2*b)**2
    E_ratio = np.maximum(0.0, (Q_local / caudal_por_tobera_m3s) - 1.0) * 0.5 
    T_nucleo = (T_impulsion + E_ratio * T_lateral) / (1.0 + E_ratio)
    
    perfil = np.exp(-(np.abs(X - (cx + desp_x)) / np.maximum(b, 0.02))**2)
    perfil[Y > altura_tobera] = 0
    
    V_vel_total += (Uc * perfil)
    Influencia_total = np.maximum(Influencia_total, perfil) 
    
    T_num += (T_nucleo * perfil)
    T_den += (perfil + 1e-6)

Influencia_total = np.clip(Influencia_total, 0.0, 1.0)
T_core = T_num / T_den
T = T_lateral - (T_lateral - T_core) * Influencia_total
T = np.clip(T, T_impulsion, T_lateral)

V_mag = np.sqrt(V_lateral**2 + V_vel_total**2)

# --- 3. GRÁFICAS ---
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plt.suptitle(f"EVALUACIÓN EN CALMA: Caudal {caudal_total_m3h} m3/h | Viento {V_lateral} m/s", fontweight='bold')

im_T = axs[0].imshow(T, origin='lower', extent=[x.min(), x.max(), 0, Ly], cmap='coolwarm', vmin=20, vmax=45)
axs[0].set_title("Campo Térmico en Vacío (Alta Resolución)")
axs[0].set_xlabel("Distancia transversal (m)"); axs[0].set_ylabel("Altura (m)")
plt.colorbar(im_T, ax=axs[0], label="Temperatura (ºC)")
for cx in centros_toberas: axs[0].add_patch(Rectangle((cx - D0/2, altura_tobera), D0, 0.05, color='darkgray'))

im_V = axs[1].imshow(V_mag, origin='lower', extent=[x.min(), x.max(), 0, Ly], cmap='viridis', vmin=0, vmax=3.0)
axs[1].set_title("Campo de Velocidad")
axs[1].set_xlabel("Distancia transversal (m)"); axs[1].set_ylabel("Altura (m)")
plt.colorbar(im_V, ax=axs[1], label="Velocidad (m/s)")
for cx in centros_toberas: axs[1].add_patch(Rectangle((cx - D0/2, altura_tobera), D0, 0.05, color='darkgray'))

plt.tight_layout()
plt.show()

# --- 4. EXTRACCIÓN DE DATOS ---
def get_temp(altura):
    idx_y = np.argmin(np.abs(y - altura))
    idx_x = np.argmin(np.abs(x - 0.0))
    return T[idx_y, idx_x]

print("="*70)
print(f" RESULTADOS TÉCNICOS (Viento en calma: {V_lateral} m/s)")
print("="*70)
print(f"-> Temp. Cabeza (1.70m) : {get_temp(1.70):.1f} ºC")
print(f"-> Temp. Tronco (1.10m) : {get_temp(1.10):.1f} ºC")
print(f"-> Temp. Tobillo (0.10m): {get_temp(0.10):.1f} ºC")
print(f"-> Gradiente (Cab-Tob)  : {get_temp(1.70) - get_temp(0.10):+.1f} ºC")
print("="*70)

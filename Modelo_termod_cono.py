import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# =========================================================
# 🎛️ PANEL DE CONTROL INTUITIVO
# =========================================================

# --- 1. ENTORNO DE LA NAVE ---
T_lateral = 45.0      
V_lateral = 2.0       # Velocidad del viento transversal (m/s)

# --- 2. SISTEMA DE REFRIGERACIÓN BASE ---
T_impulsion = 20.0    
V_impulsion_objetivo = 3.0  # m/s de salida deseada
caudal_total_base = 900.0   # m3/h combinados para los 3 difusores
distancia_conos = 1.5       # Metros de separación entre centros

# --- 3. GEOMETRÍA DEL CONO ---
angulo_apertura = 30.0      # Ángulo de apertura total de la tobera (grados)

# --- 4. MEJORA DEL SISTEMA ---
activar_ventilador_extra = False  
aumento_caudal_porcentaje = 100.0  # 100% = el doble de caudal (1800 m3/h)
estrategia_ventilador = "area"     # "area" (agranda la boca) o "velocidad" (sopla más fuerte)

# =========================================================
# ⚙️ MOTOR DE INGENIERÍA Y RESUMEN
# =========================================================

caudal_base_m3s = caudal_total_base / 3600.0
caudal_por_cono_m3s = caudal_base_m3s / 3.0
area_original = caudal_por_cono_m3s / V_impulsion_objetivo
radio_original = np.sqrt(area_original / np.pi)

V_real = V_impulsion_objetivo
radio_real = radio_original
caudal_final_m3h = caudal_total_base

if activar_ventilador_extra:
    factor = 1.0 + (aumento_caudal_porcentaje / 100.0)
    caudal_final_m3h = caudal_total_base * factor
    if estrategia_ventilador == "velocidad":
        V_real = V_impulsion_objetivo * factor
    elif estrategia_ventilador == "area":
        radio_real = radio_original * np.sqrt(factor)

caudal_final_por_cono = caudal_final_m3h / 3.0

print(f"--- PARÁMETROS REALES EN SIMULACIÓN ---")
print(f"Viento lateral: {V_lateral} m/s a {T_lateral} ºC")
print(f"Radio físico de cada tobera: {radio_real*100:.1f} cm")
print(f"Velocidad efectiva de salida: {V_real:.2f} m/s")
print(f"Ángulo de descarga: {angulo_apertura}º")
print(f"Caudal inyectado por difusor: {caudal_final_por_cono:.0f} m3/h")
print(f"CAUDAL TOTAL DEL SISTEMA (3 difusores): {caudal_final_m3h:.0f} m3/h")
print(f"---------------------------------------")

# =========================
# 📐 DOMINIO Y CONSTANTES
# =========================

altura_cono = 2.6
T_corporal = 37.0

Lx = 8.0
Ly = 3.0
dx = 0.04
dy = 0.04
nx = int(Lx/dx)
ny = int(Ly/dy)
x = np.linspace(-4, 4, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

centros_conos = [-distancia_conos, 0.0, distancia_conos]

# =========================
# 🧍 PERSONA (Obstáculo)
# =========================

persona_x = 0.0
ancho_cuerpo = 0.5
altura_cuerpo = 1.5
radio_cabeza = 0.2
mask_persona = np.zeros_like(X, dtype=bool)

for j in range(ny):
    for i in range(nx):
        if (abs(x[i]-persona_x) < ancho_cuerpo/2 and y[j] < altura_cuerpo):
            mask_persona[j,i] = True
        dx_h = x[i] - persona_x
        dy_h = y[j] - (altura_cuerpo + radio_cabeza)
        if np.sqrt(dx_h**2 + dy_h**2) < radio_cabeza:
            mask_persona[j,i] = True

# =========================
# 🌪️ CAMPO DE VELOCIDAD ACÚSTICO-GEOMÉTRICO
# =========================

U = np.zeros_like(X)
V = np.zeros_like(X)

U += V_lateral * np.exp(-((Y-1.2)/1.2)**2)

for i in range(nx):
    for j in range(ny):
        if Y[j,i] <= altura_cono:
            vel_vertical = 0
            for cx in centros_conos:
                distancia_vertical = altura_cono - Y[j,i]
                
                expansion = distancia_vertical * np.tan(np.radians(angulo_apertura / 2.0))
                radio_local = radio_real + expansion
                r = abs(X[j,i] - cx)
                
                if r <= radio_local * 1.5:
                    perfil = np.exp(-(r / radio_local)**2)
                    decay = (radio_real / radio_local)
                    vel_vertical += V_real * decay * perfil
                    
            V[j,i] -= vel_vertical

        if not mask_persona[j,i]:
            dx_p = x[i] - persona_x
            dist_cabeza = np.sqrt(dx_p**2 + (y[j] - (altura_cuerpo + radio_cabeza))**2)
            if dist_cabeza < 0.6:
                U[j,i] += V_real * 0.3 * np.sign(dx_p) * np.exp(-dist_cabeza/0.2)
            if dx_p > 0 and dx_p < 2.5 and y[j] < altura_cuerpo + 0.5:
                U[j,i] += V_lateral * 0.7 * np.exp(-dx_p/1.2)
                V[j,i] -= V_real * 0.15 * np.exp(-dx_p/0.8)

U[mask_persona] = 0.0
V[mask_persona] = 0.0
V_mag = np.sqrt(U**2 + V**2)

# =========================
# 🌡️ MODELO TERMODINÁMICO ACOPLADO
# =========================

T = np.ones_like(X) * T_lateral
steps = 350  
k_dif = 0.010  
k_adv = 0.010  

for step in range(steps):
    T_new = T.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            if mask_persona[j,i]:
                T_new[j,i] = T_corporal
                continue
            
            if U[j,i] > 0: dTdx = (T[j,i] - T[j,i-1]) / dx
            else: dTdx = (T[j,i+1] - T[j,i]) / dx
            if V[j,i] > 0: dTdy = (T[j,i] - T[j-1,i]) / dy
            else: dTdy = (T[j+1,i] - T[j,i]) / dy
                
            adveccion = -k_adv * (U[j,i]*dTdx + V[j,i]*dTdy)
            difusion = k_dif * (T[j+1,i] + T[j-1,i] + T[j,i+1] + T[j,i-1] - 4*T[j,i])
            T_new[j,i] = T[j,i] + adveccion + difusion
            
            if y[j] <= altura_cono:
                for cx in centros_conos:
                    distancia_vertical = altura_cono - y[j]
                    expansion = distancia_vertical * np.tan(np.radians(angulo_apertura / 2.0))
                    radio_local = radio_real + expansion
                    r = abs(x[i] - cx)
                    
                    if r <= radio_local:
                        peso_velocidad = min(abs(V[j,i]) / V_real, 1.0)
                        T_new[j,i] = T_new[j,i] * (1 - peso_velocidad * 0.15) + T_impulsion * (peso_velocidad * 0.15)
                        
    for j in range(ny):
        if y[j] > altura_cono:
            T_new[j,:] = T_lateral
            
    T = T_new

# =========================
# 🎨 VISUALIZACIÓN A ESCALA REAL
# =========================

fig, axs = plt.subplots(2, 2, figsize=(16, 8))

im_T = axs[0,0].imshow(
    T, origin='lower', extent=[x.min(), x.max(), 0, Ly],
    cmap='coolwarm', vmin=20, vmax=45
)
axs[0,0].streamplot(X, Y, U, V, color=(1.0, 1.0, 1.0, 0.5), density=0.8, linewidth=0.5)
axs[0,0].add_patch(Rectangle((persona_x - ancho_cuerpo/2, 0), ancho_cuerpo, altura_cuerpo, color='black'))
axs[0,0].add_patch(Circle((persona_x, altura_cuerpo + radio_cabeza), radio_cabeza, color='black'))
cbar_T = plt.colorbar(im_T, ax=axs[0,0])
cbar_T.set_label("Temperatura (ºC)")
axs[0,0].set_title(f"Campo Térmico (Apertura: {angulo_apertura}º)")
axs[0,0].set_xlabel("Distancia horizontal (m)")
axs[0,0].set_ylabel("Altura (m)")
axs[0,0].set_aspect('equal') 

im_V = axs[0,1].imshow(
    V_mag, origin='lower', extent=[x.min(), x.max(), 0, Ly],
    cmap='viridis', vmin=0, vmax=3.5
)
axs[0,1].streamplot(X, Y, U, V, color='white', density=1.2, linewidth=1)
axs[0,1].add_patch(Rectangle((persona_x - ancho_cuerpo/2, 0), ancho_cuerpo, altura_cuerpo, color='black'))
axs[0,1].add_patch(Circle((persona_x, altura_cuerpo + radio_cabeza), radio_cabeza, color='black'))
cbar_V = plt.colorbar(im_V, ax=axs[0,1])
cbar_V.set_label("Velocidad (m/s)")
axs[0,1].set_title("Campo de Velocidades Geométrico")
axs[0,1].set_xlabel("Distancia horizontal (m)")
axs[0,1].set_ylabel("Altura (m)")
axs[0,1].set_aspect('equal') 

altura_eval = 1.1
idx_y = np.argmin(np.abs(y - altura_eval))
vel_linea = V_mag[idx_y, :]
temp_linea = T[idx_y, :]

axs[1,0].plot(x, temp_linea, color='red')
axs[1,0].axvline(-0.5, color='gray', linestyle='--')
axs[1,0].axvline(0.5, color='gray', linestyle='--')
axs[1,0].axhspan(20, 26, color='green', alpha=0.1, label='Confort (20-26ºC)')
axs[1,0].set_title(f"Perfil de Temperatura a 1.1 m")
axs[1,0].set_xlabel("Distancia horizontal (m)")
axs[1,0].set_ylabel("Temperatura (ºC)")
axs[1,0].grid(True)
axs[1,0].set_xlim([x.min(), x.max()])

axs[1,1].plot(x, vel_linea, color='blue')
axs[1,1].axvline(-0.5, color='gray', linestyle='--')
axs[1,1].axvline(0.5, color='gray', linestyle='--')
axs[1,1].axhline(0.2, color='orange', linestyle='--', label='Ráfagas (0.2 m/s)')
axs[1,1].set_title(f"Perfil de Velocidad a 1.1 m")
axs[1,1].set_xlabel("Distancia horizontal (m)")
axs[1,1].set_ylabel("Velocidad (m/s)")
axs[1,1].grid(True)
axs[1,1].set_xlim([x.min(), x.max()])

plt.tight_layout()
plt.show()

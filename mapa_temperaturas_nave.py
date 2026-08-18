%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# MODELO TERMO-AERODINAMICO REDUCIDO V7.2 (DEFINITIVO)
# ============================================================

# ============================================================
# 1. PANEL DE CONTROL EDITABLE
# ============================================================
T_exterior = 40.0         # Temperatura exterior (ºC)
T_base_interior = 39.0    # Temperatura base interior nave (ºC)
T_impulsion = 20.0        # Temperatura del aire en la tobera (ºC)
T_vehiculo = 80.0         # Temperatura superficial de los vehículos (ºC)
T_cubierta = 42.0         # Temperatura radiante de cubierta (ºC)

# Condiciones de contorno (Aire en calma / Puertas abiertas)
velocidad_viento_exterior = 0.0  # v = 0 m/s
estado_puertas = "Abiertas (Ventilación reglamentaria R.D. 486/1997)"

# Geometría y Operativa
Q_UTA_m3h = 4000.0        # Caudal total de la UTA (m3/h)
tipo_difusor = 'ELIPTICO' # Opciones: 'CIRCULAR', 'ELIPTICO', 'LINEAL'

# ============================================================
# 2. GEOMETRIA DE LA NAVE Y ROI (ZONA A1)
# ============================================================

L_nave = 42.0
W_nave = 6.50

z_impulsion = 2.60
z_mapa = 1.10
distancia_vertical = z_impulsion - z_mapa

# Delimitación geométrica de la Zona A1 / Puesto de trabajo (ROI)
roi_x_min, roi_x_max = 1.5, 5.5
roi_y_min, roi_y_max = 4.5, 6.5

# ============================================================
# 3. DARDOS DE IMPULSION Y VEHICULOS
# ============================================================

N_dardos_total = 9
diametro_dardo = 0.1881  # Diámetro equivalente nominal (m) -> 188.1 mm

dardos = np.array([
    [2.00, 5.80],
    [2.75, 5.80],
    [3.50, 5.80],
    [4.25, 5.80],
    [5.00, 5.80],
    [13.00, 5.80],
    [16.00, 5.80],
    [36.00, 5.80]
], dtype=float)

vehiculos = np.array([
    [4.00, 3.00],
    [15.00, 3.00],
    [30.00, 3.00]
], dtype=float)

vehiculo_L = 4.50
vehiculo_W = 2.00

# ============================================================
# 4. CALCULO HIDRODINAMICO Y TERMICO
# ============================================================

rho = 1.20          # kg/m3
mu = 1.81e-5        # Pa.s
g = 9.81            # m/s2
beta_vol = 1.0 / (273.15 + T_base_interior)

Q_por_dardo_m3h = Q_UTA_m3h / N_dardos_total
Q_por_dardo = Q_por_dardo_m3h / 3600.0
A_dardo = np.pi * diametro_dardo**2 / 4.0
V_salida = Q_por_dardo / A_dardo

Re = (rho * V_salida * diametro_dardo) / mu

if tipo_difusor == 'CIRCULAR':
    r0 = diametro_dardo / 2.0
    Kb = 0.10
    r_mapa = r0 + Kb * distancia_vertical
    sigma_x = max(r_mapa / 1.5, 0.08)
    sigma_y = max(r_mapa / 1.5, 0.08)
    A_mapa = np.pi * r_mapa**2
    beta_m = r_mapa / r0

elif tipo_difusor == 'ELIPTICO':
    AR_eliptico = 3.0
    b0 = diametro_dardo / (2.0 * np.sqrt(AR_eliptico))
    a0 = AR_eliptico * b0
    Kb_a = 0.08
    Kb_b = 0.12
    a_mapa = a0 + Kb_a * distancia_vertical
    b_mapa = b0 + Kb_b * distancia_vertical
    sigma_x = max(b_mapa / 1.5, 0.08)
    sigma_y = max(a_mapa / 1.5, 0.08)
    A_mapa = np.pi * a_mapa * b_mapa
    beta_m = np.sqrt(A_mapa / A_dardo)

elif tipo_difusor == 'LINEAL':
    b0_lineal = 0.02
    L0_lineal = A_dardo / b0_lineal
    Kb_b = 0.11
    Kb_L = 0.02
    b_mapa = b0_lineal + 2.0 * Kb_b * distancia_vertical
    L_mapa = L0_lineal + 2.0 * Kb_L * distancia_vertical
    sigma_x = max(b_mapa / 2.0, 0.05)
    sigma_y = max(L_mapa / 2.0, 0.20)
    A_mapa = L_mapa * b_mapa
    beta_m = np.sqrt(A_mapa / A_dardo)

delta_T0 = T_base_interior - T_impulsion
d_eq = np.sqrt(4.0 * A_dardo / np.pi)
Ar = (g * beta_vol * delta_T0 * d_eq) / (V_salida**2)
f_buoyancy = 1.0 / (1.0 + 0.35 * np.sqrt(Ar) * (distancia_vertical / d_eq))
fraccion_entrainment = float(np.clip(1.0 - (f_buoyancy / beta_m), 0.0, 0.98))
T_mezcla = T_impulsion + fraccion_entrainment * (T_base_interior - T_impulsion)

# ============================================================
# 5. MALLA Y CAMPOS ESCALARES
# ============================================================

nx = 421
ny = 131
x = np.linspace(0.0, L_nave, nx)
y = np.linspace(0.0, W_nave, ny)
X, Y = np.meshgrid(x, y)

T = np.full_like(X, T_base_interior, dtype=float)

for xd, yd in dardos:
    R2_aniso = ((X - xd) / sigma_x)**2 + ((Y - yd) / sigma_y)**2
    w = np.exp(-0.5 * R2_aniso)
    T -= (T_base_interior - T_mezcla) * w

puertas_x = [0.0, L_nave]
for xp in puertas_x:
    wx = np.exp(-0.5 * ((X - xp) / 2.0)**2)
    wy = np.exp(-0.5 * ((Y - W_nave / 2.0) / 2.0)**2)
    T += 1.0 * (T_exterior - T_base_interior) * wx * wy

T_mr = np.full_like(X, T_base_interior + 3.0, dtype=float)

for xc, yc in vehiculos:
    inside = ((np.abs(X - xc) <= vehiculo_L / 2.0) & (np.abs(Y - yc) <= vehiculo_W / 2.0))
    dx_rect = np.maximum(np.abs(X - xc) - vehiculo_L / 2.0, 0.0)
    dy_rect = np.maximum(np.abs(Y - yc) - vehiculo_W / 2.0, 0.0)
    distancia_rectangulo = np.sqrt(dx_rect**2 + dy_rect**2)
    w_vehicle = np.exp(-0.5 * (distancia_rectangulo / 1.10)**2)
    w_vehicle = np.where(inside, 1.0, w_vehicle)
    T += 2.5 * w_vehicle

    z_veh_diff = 0.30
    dist_3d = np.sqrt((X - xc)**2 + (Y - yc)**2 + z_veh_diff**2)
    w_rad = z_veh_diff / (2.0 * np.pi * dist_3d**3)
    w_rad_norm = w_rad / np.max(w_rad)
    T_mr += 6.0 * w_rad_norm

T_op = 0.5 * T + 0.5 * T_mr
T_max_permitida = max(T_base_interior + 10.0, T_vehiculo)
T = np.clip(T, T_impulsion, T_max_permitida)
T_op = np.clip(T_op, T_impulsion, T_max_permitida)

# ============================================================
# 6. EXTRACCION DE ESTADISTICAS
# ============================================================

mask_roi = (X >= roi_x_min) & (X <= roi_x_max) & (Y >= roi_y_min) & (Y <= roi_y_max)

T_media_nave = float(np.mean(T))
Top_media_nave = float(np.mean(T_op))

T_media_zona_A1 = float(np.mean(T[mask_roi]))
Top_media_zona_A1 = float(np.mean(T_op[mask_roi]))
cob_35_zona_A1 = 100.0 * np.mean(T[mask_roi] <= 35.0)

# ============================================================
# 7. INFORME DETALLADO EN CONSOLA (CON DIAMETRO DE TOBERA)
# ============================================================

print("=" * 82)
print(" MODELO TERMO-AERODINAMICO REDUCIDO V7.2 (DEFINITIVO)")
print("=" * 82)
print(f"--- CONDICIONES DE CONTORNO CONFIGURADAS ---")
print(f"Geometría de difusor            : {tipo_difusor}")
print(f"Diámetro equivalente de tobera  : {diametro_dardo * 1000:8.1f} mm ({diametro_dardo:.4f} m)")
print(f"Temperatura exterior            : {T_exterior:8.2f} ºC")
print(f"Temperatura impulsión toberas   : {T_impulsion:8.2f} ºC")
print(f"Velocidad salida toberas (V)    : {V_salida:8.3f} m/s")
print(f"Velocidad viento exterior (v)   : {velocidad_viento_exterior:8.2f} m/s (Aire en calma)")
print(f"Estado de portones              : {estado_puertas}")

print(f"\n--- RESULTADOS METRICOS ---")
print(f"Temperatura media de la nave    : {T_media_nave:8.2f} ºC")
print(f"Temperatura media Zona A1 (ROI) : {T_media_zona_A1:8.2f} ºC")
print(f"Temperatura Operativa Zona A1   : {Top_media_zona_A1:8.2f} ºC")
print(f"Cobertura aire <= 35 ºC (Zona A1): {cob_35_zona_A1:8.1f} %")
print("=" * 82)

# ============================================================
# 8. REPRESENTACION GRAFICA
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5))

levels = np.linspace(np.floor(T.min()), np.ceil(T.max()), 25)
cf = ax.contourf(X, Y, T, levels=levels, cmap="RdYlBu_r")

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("Temperatura del Aire (ºC)")

cs_op = ax.contour(X, Y, T_op, levels=[32.0, 35.0], colors=["blue", "darkorange"],
                   linewidths=1.2, linestyles="--")
ax.clabel(cs_op, inline=True, fontsize=8, fmt=r"Top=%1.0fºC")

ax.scatter(dardos[:, 0], dardos[:, 1], marker="o", s=40, facecolors="cyan",
           edgecolors="black", label=f"Difusores ({tipo_difusor}, D={diametro_dardo*1000:.1f}mm)")

for i, (xc, yc) in enumerate(vehiculos, 1):
    rect = Rectangle((xc - vehiculo_L / 2.0, yc - vehiculo_W / 2.0),
                     vehiculo_L, vehiculo_W, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xc, yc, f"V{i}", ha="center", va="center", fontsize=8, fontweight="bold")

roi_rect = Rectangle((roi_x_min, roi_y_min), roi_x_max - roi_x_min, roi_y_max - roi_y_min,
                     fill=False, edgecolor="green", linewidth=2.0, linestyle=":", label="Zona A1 (ROI Puesto Trabajo)")
ax.add_patch(roi_rect)

ax.axvline(0.0, color="red", linewidth=2.0)
ax.axvline(L_nave, color="red", linewidth=2.0)

ax.set_xlim(0, L_nave)
ax.set_ylim(0, W_nave)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Modelo V7.2 | Difusor: {tipo_difusor} (D={diametro_dardo*1000:.1f}mm) | T_ext={T_exterior}ºC", fontsize=11)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=10, frameon=True)

plt.tight_layout()
plt.show()

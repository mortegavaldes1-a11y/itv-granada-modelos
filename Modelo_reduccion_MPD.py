import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# PARÁMETROS BASE Y CALIBRACIÓN DEL MODELO
# =====================================================================
C_ext_calibracion = 35.0
C_nave_calibracion = 6.0
C_interna_vehiculos = 1.5 # Fondo propio generado por los vehículos en la nave
k_dilucion = (C_nave_calibracion - C_interna_vehiculos) / C_ext_calibracion

# Jornada laboral de 7 horas (420 minutos) y parámetros del cono
horas_jornada = 7.0
F_tiempo_cono = 0.75      # 75% del tiempo bajo el cono de la UTA
FI = 0.15                 # Factor de Infiltración: 15% de mezcla lateral

# Eje X: Porcentaje de eficiencia de filtración de la UTA (de 0% a 100%)
eficiencia_pct = np.linspace(0, 100, 200)
eficiencia_decimal = eficiencia_pct / 100.0

# Escenarios de concentración exterior de MPD a representar
escenarios_ext = {
    "Zona Sierra / Muy Limpia (15 µg/m³)": 15.0,
    "Media Estándar ITV (35 µg/m³)": 35.0,
    "Zona Urbana Saturada (50 µg/m³)": 50.0,
    "Polígono Industrial Crítico (75 µg/m³)": 75.0
}

# =====================================================================
# MOTOR DE CÁLCULO Y GENERACIÓN DE LA GRÁFICA
# =====================================================================
plt.figure(figsize=(13, 7))
colores = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

for (nombre_escenario, C_ext), color in zip(escenarios_ext.items(), colores):
    
    # 1. Concentración macroclimática en la nave para este exterior
    C_nave_macroclima = C_interna_vehiculos + k_dilucion * C_ext
    
    # 2. Calcular la exposición TWA(7h) para cada porcentaje de filtrado
    twa_7h_list = []
    for eta in eficiencia_decimal:
        C_impulsion = C_ext * (1.0 - eta)
        C_respiracion = C_impulsion * (1.0 - FI) + C_nave_macroclima * FI
        E_TWA = C_respiracion * F_tiempo_cono + C_nave_macroclima * (1.0 - F_tiempo_cono)
        twa_7h_list.append(E_TWA)
        
    # Dibujar la curva para este escenario
    plt.plot(eficiencia_pct, twa_7h_list, label=nombre_escenario, color=color, linewidth=2.5)

# =====================================================================
# CONFIGURACIÓN Y ESTÉTICA DE LA GRÁFICA
# =====================================================================
plt.axhline(50, color='black', linestyle='-.', linewidth=1.5, label="VLA-ED INSST (50 µg/m³)")

plt.axvline(5, color='gray', linestyle=':', alpha=0.6)
plt.text(6, 2, "G4 (5%)", color='gray', fontsize=9)

plt.axvline(75, color='gray', linestyle=':', alpha=0.6)
plt.text(76, 2, "F8 (75%)", color='gray', fontsize=9)

plt.axvline(90, color='gray', linestyle=':', alpha=0.6)
plt.text(91, 2, "F9 (90%)", color='gray', fontsize=9)

plt.axvline(99, color='gray', linestyle=':', alpha=0.6)
plt.text(95, 12, "ESP/Nanofibras (99%)", color='gray', fontsize=9)

plt.title(f"Evolución de la Exposición TWA ({horas_jornada}h) del Inspector vs. Eficiencia de Filtración\n(Agrupado por diferentes niveles de contaminación exterior)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Eficiencia de Filtración de la UTA frente a MPD (%)", fontsize=12)
plt.ylabel(f"Exposición Efectiva Ponderada TWA ({horas_jornada}h) (µg/m³)[cite: 1]", fontsize=12)

plt.xlim(0, 100)
plt.ylim(0, 60)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
plt.tight_layout()

plt.show()

!pip install psychrolib

import numpy as np
import matplotlib.pyplot as plt
import psychrolib

# ============================================================
# CONFIGURACIÓN TERMODINÁMICA Y PSICROMÉTRICA
# ============================================================

psychrolib.SetUnitSystem(psychrolib.SI)

# Presión atmosférica estimada para la altitud de Granada (aprox. 738 m)
P_ATM = 92900.0  # Pa

# Vectorizamos las funciones de psychrolib para procesar arrays de NumPy
get_w = np.vectorize(lambda t, hr: psychrolib.GetHumRatioFromRelHum(t, hr/100.0, P_ATM))
get_twb = np.vectorize(lambda t, w: psychrolib.GetTWetBulbFromHumRatio(t, w, P_ATM))
get_h = np.vectorize(lambda t, w: psychrolib.GetMoistAirEnthalpy(t, w))
get_rho = np.vectorize(lambda t, w: psychrolib.GetMoistAirDensity(t, w, P_ATM))
get_hr_from_w = np.vectorize(lambda t, w: psychrolib.GetRelHumFromHumRatio(t, w, P_ATM) * 100.0)
get_w_from_h_t = np.vectorize(lambda h, t: psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h, t))
get_wsat = np.vectorize(lambda t: psychrolib.GetSatHumRatio(t, P_ATM))

# ============================================================
# 1. PARÁMETROS DEL DÍA TIPO Y UTA
# ============================================================

T_max = 45.0          # ºC
T_min = 28.0          # ºC
HR_min = 15.0         # %
HR_max = 40.0         # %
hora_pico = 17.0      # 17:00
hora_inicio = 7
hora_fin = 21

T_impulsion = 20.0    # ºC
caudal_vol = 4000.0   # m3/h

# Eficiencias técnicas de los paneles evaporativos
EFICIENCIA_DIRECTA = 0.85
EFICIENCIA_INDIRECTA = 0.65

# ============================================================
# 2. FUNCIONES DE PERFIL Y CONTROL
# ============================================================

def perfil_temperatura(horas):
    fase = 2.0 * np.pi * (horas - hora_pico) / 24.0
    return ((T_max + T_min) / 2.0 + (T_max - T_min) / 2.0 * np.cos(fase))

def perfil_humedad(horas):
    fase = 2.0 * np.pi * (horas - hora_pico) / 24.0
    return ((HR_max + HR_min) / 2.0 - (HR_max - HR_min) / 2.0 * np.cos(fase))

def control_evaporativo(T_ext):
    factor = np.zeros_like(T_ext, dtype=float)
    factor[(T_ext >= 30.0) & (T_ext < 35.0)] = 0.50
    factor[(T_ext >= 35.0) & (T_ext < 38.0)] = 0.75
    factor[T_ext >= 38.0] = 1.00
    return factor

def COP(T_ext):
    cop = 5.5 - 0.07 * (T_ext - 25.0)
    return np.clip(cop, 2.5, 5.5)

# ============================================================
# 3. MODELOS PSICROMÉTRICOS DE LOS EVAPORATIVOS
# ============================================================

def evaporativo_directo(T_ext, W_ext, factor):
    Twb = get_twb(T_ext, W_ext)
    T_out = T_ext - EFICIENCIA_DIRECTA * factor * (T_ext - Twb)
    h_in = get_h(T_ext, W_ext)
    W_out = np.where(factor > 0, get_w_from_h_t(h_in, T_out), W_ext)
    return T_out, W_out

def evaporativo_indirecto(T_ext, W_ext, factor):
    Twb = get_twb(T_ext, W_ext)
    T_out = T_ext - EFICIENCIA_INDIRECTA * factor * (T_ext - Twb)
    W_out = W_ext.copy()
    return T_out, W_out

def evaporativo_doble(T_ext, W_ext, factor):
    T_int, W_int = evaporativo_indirecto(T_ext, W_ext, factor)
    Twb_int = get_twb(T_int, W_int)
    T_out = T_int - EFICIENCIA_DIRECTA * factor * (T_int - Twb_int)
    h_int = get_h(T_int, W_int)
    W_out = np.where(factor > 0, get_w_from_h_t(h_int, T_out), W_int)
    return T_out, W_out

# ============================================================
# 4. EJECUCIÓN DEL MODELO HORARIO
# ============================================================

horas = np.arange(0.0, 24.0, 0.25)
T_ext = perfil_temperatura(horas)
HR_ext = perfil_humedad(horas)
W_ext = get_w(T_ext, HR_ext)
factor_evap = control_evaporativo(T_ext)

T_ninguno, W_ninguno = T_ext.copy(), W_ext.copy()
T_directo, W_directo = evaporativo_directo(T_ext, W_ext, factor_evap)
T_indirecto, W_indirecto = evaporativo_indirecto(T_ext, W_ext, factor_evap)
T_doble, W_doble = evaporativo_doble(T_ext, W_ext, factor_evap)

escenarios = {
    "Sin evaporativo": {"T": T_ninguno, "W": W_ninguno},
    "Evaporativo directo": {"T": T_directo, "W": W_directo},
    "Evaporativo indirecto": {"T": T_indirecto, "W": W_indirecto},
    "Evaporativo dos etapas": {"T": T_doble, "W": W_doble}
}

# ============================================================
# 5. CÁLCULO DE POTENCIA RIGUROSO (SENSIBLE + LATENTE)
# ============================================================

W_sat_impulsion = get_wsat(np.full_like(horas, T_impulsion))

for nombre, datos in escenarios.items():
    T_in = datos["T"]
    W_in = datos["W"]
    
    rho = get_rho(T_in, W_in)
    m_dot_kg_s = (caudal_vol / 3600.0) * rho
    
    W_impulsion = np.minimum(W_in, W_sat_impulsion)
    
    h_in = get_h(T_in, W_in)
    h_impulsion = get_h(np.full_like(T_in, T_impulsion), W_impulsion)
    
    Q_frio = m_dot_kg_s * np.maximum(0.0, (h_in - h_impulsion) / 1000.0)
    
    funcionando = (horas >= hora_inicio) & (horas <= hora_fin)
    Q_frio[~funcionando] = 0.0
    
    cop = COP(T_ext)
    P_electrica = np.zeros_like(Q_frio)
    P_electrica[funcionando] = Q_frio[funcionando] / cop[funcionando]
    
    datos["Q_frio"] = Q_frio
    datos["P_electrica"] = P_electrica
    datos["HR_out"] = get_hr_from_w(T_in, W_in)

# ============================================================
# 6. RESUMEN Y ANÁLISIS DE RESULTADOS
# ============================================================

mask = (horas >= hora_inicio) & (horas <= hora_fin)

print("=" * 70)
print("        RESUMEN TERMODINÁMICO DE ESCENARIOS")
print("=" * 70)

for nombre, datos in escenarios.items():
    Q = datos["Q_frio"][mask]
    P = datos["P_electrica"][mask]
    
    energia_frio = np.sum(Q) * 0.25
    energia_electrica = np.sum(P) * 0.25
    
    print(f"\n--- {nombre} ---")
    print(f"Potencia frigorífica máxima: {np.max(Q):.2f} kW")
    print(f"Potencia eléctrica máxima:   {np.max(P):.2f} kW")
    print(f"Energía eléctrica diaria:    {energia_electrica:.2f} kWh")
    print(f"HR máxima entrada UTA:       {np.max(datos['HR_out'][mask]):.1f} %")

print("\n" + "=" * 70)

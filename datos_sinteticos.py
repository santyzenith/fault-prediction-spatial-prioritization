"""Genera un CSV sintético con el mismo formato que los datos reales, que son privados.
Sirve para correr el pipeline completo; los resultados no son los del artículo.

Uso: uv run datos_sinteticos.py   (escribe datasets/sintetico.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

N = 1000
rng = np.random.default_rng(42)

carga = rng.lognormal(mean=4.5, sigma=1.2, size=N).round(1)
distancia = rng.uniform(100, 30000, N)
mantenimiento = rng.choice([0.0, 1.0], N, p=[0.9, 0.1])
vegetacion = rng.uniform(0, 1, N).round(3)

# La falla depende de la carga, la distancia y el mantenimiento, más ruido
logit = 1.2 * (np.log(carga) - 4.5) + 0.00004 * distancia + 1.5 * mantenimiento - 1.4
falla = rng.uniform(size=N) < 1 / (1 + np.exp(-logit))

df = pd.DataFrame({
    "Falla": np.where(falla, "Falla", "Sin Falla"),
    "Proteccion": np.where(falla, "Fusible", "Sin Proteccion"),
    "Linea": np.where(falla, "Alimentador", "Sin Linea de Falla"),
    "Causa": np.where(falla, "Vegetacion", "No Existe Causa"),
    "Carga_Fuer": carga,
    "Fecha_de_I": np.nan, "Hora_de_In": np.nan, "Fecha_de_F": np.nan,
    "Hora_de_Fi": np.nan, "Duracion_F": np.nan,
    "X": rng.integers(500000, 530000, N),
    "Y": rng.integers(9000000, 9030000, N),
    "Riesgo": falla.astype(int),
    "DistanciaFuente": distancia,
    "RasterMantenimiento_1": mantenimiento,
    "DensidadVegetacion": vegetacion,
    "DescargarAtmosKm2/ano": rng.choice([6, 7], N),
})
df.loc[rng.choice(N, 10, replace=False), "DensidadVegetacion"] = np.nan

destino = Path(__file__).parent / "datasets" / "sintetico.csv"
destino.parent.mkdir(exist_ok=True)
df.to_csv(destino, index=False)
print(f"{destino}: {N} registros, {falla.mean():.0%} con falla")

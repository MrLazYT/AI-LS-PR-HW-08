import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfs = []
# Створення 365 днів (1 рік)
days = np.arange(-182, 183)

for i in range(3, 4):
    dates = pd.date_range(start=f"202{i}-01-01", periods=365)

    # Обчислення споживання
    consumption = 0.1 * (days ** 2) * 0.01 + 23
    
    # Побудова датафрейму
    df = pd.DataFrame({
        "date": dates,
        "consumption_kwh": consumption.round(2)
    })

    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Збереження у CSV
df_all.to_csv('daily_energy_consumption.csv', index=False)
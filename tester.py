import numpy as np
import matplotlib.pyplot as plt

# Генерация данных для диапазона от 250000 до 500000 шагов
total_timesteps = np.linspace(250000, 500000, 126)  # 126 точек для имитации итераций

# Генерация синтетических данных, отражающих наблюдаемое поведение

# Средняя награда: улучшение до определенного момента, затем застревание в локальном минимуме
ep_rew_mean = -115 + 25 * (1 - np.exp(-(total_timesteps - 250000) / 50000)) + \
              5 * np.sin((total_timesteps - 320000) / 40000) + \
              np.random.normal(0, 1.5, len(total_timesteps))

# Стандартное отклонение: монотонное уменьшение до критически низких значений
std_values = 0.085 * np.exp(-(total_timesteps - 250000) / 120000) + \
             0.035 + np.random.normal(0, 0.001, len(total_timesteps))
std_values = np.maximum(std_values, 0.032)  # Ограничение минимального значения

# Энтропия: парадоксальный рост при уменьшении std
entropy_loss = 3.8 + 1.6 * ((total_timesteps - 250000) / 250000) + \
               np.random.normal(0, 0.08, len(total_timesteps))

# Explained variance: высокие значения с небольшими колебаниями
explained_variance = 0.85 + 0.13 * np.random.rand(len(total_timesteps)) + \
                     0.02 * np.sin((total_timesteps - 350000) / 60000)

# Value loss: стабильные низкие значения
value_loss = 0.8 * np.exp(-(total_timesteps - 250000) / 80000) + \
             0.25 + np.random.normal(0, 0.15, len(total_timesteps))
value_loss = np.maximum(value_loss, 0.1)

# Создание графиков
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Средняя награда
ax1.plot(total_timesteps, ep_rew_mean, 'b-', linewidth=2)
ax1.set_ylabel('Средняя награда за эпизод')
ax1.set_title('Динамика средней награды')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-120, -85)

# Стандартное отклонение
ax2.plot(total_timesteps, std_values, 'r-', linewidth=2)
ax2.set_ylabel('Стандартное отклонение (std)')
ax2.set_xlabel('Total Timesteps')
ax2.set_title('Динамика стандартного отклонения')
ax2.grid(True, alpha=0.3)

# Энтропия
ax3.plot(total_timesteps, entropy_loss, 'g-', linewidth=2)
ax3.set_ylabel('Entropy Loss')
ax3.set_xlabel('Total Timesteps')
ax3.set_title('Динамика энтропии')
ax3.grid(True, alpha=0.3)

# Совместный график std и энтропии
ax4.plot(total_timesteps, std_values, 'r-', linewidth=2, label='std')
ax4_twin = ax4.twinx()
ax4_twin.plot(total_timesteps, entropy_loss, 'g-', linewidth=2, label='entropy_loss')
ax4.set_xlabel('Total Timesteps')
ax4.set_ylabel('Стандартное отклонение', color='r')
ax4_twin.set_ylabel('Энтропия', color='g')
ax4.set_title('Совместная динамика std и энтропии')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Вывод основных характеристик
print("Характеристики сгенерированных данных:")
print(f"Диапазон средней награды: {np.min(ep_rew_mean):.2f} ... {np.max(ep_rew_mean):.2f}")
print(f"Диапазон стандартного отклонения: {np.min(std_values):.4f} ... {np.max(std_values):.4f}")
print(f"Диапазон энтропии: {np.min(entropy_loss):.2f} ... {np.max(entropy_loss):.2f}")
print(f"Начальное std: {std_values[0]:.4f}, конечное std: {std_values[-1]:.4f}")
print(f"Уменьшение std: {((std_values[0] - std_values[-1])/std_values[0]*100):.1f}%")

# Корреляция между std и энтропией
correlation = np.corrcoef(std_values, entropy_loss)[0, 1]
print(f"Корреляция между std и энтропией: {correlation:.3f}")
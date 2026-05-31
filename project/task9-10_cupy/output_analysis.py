import pandas as pd
import matplotlib.pyplot as plt

file_path = 'full_run.out'

df = pd.read_csv(file_path, skipinitialspace=True)

plt.figure(figsize=(8, 5))
plt.hist(df['mean_temp'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Buildings')
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.savefig("Distribution.png")

avg_mean_temp = df['mean_temp'].mean()
print(f"2. Average mean temperature: {avg_mean_temp:.2f} °C")

avg_std_temp = df['std_temp'].mean()
print(f"3. Average temperature standard deviation: {avg_std_temp:.2f} °C")

count_above_18 = (df['pct_above_18'] >= 50).sum()
print(f"4. Buildings with >= 50% of area above 18ºC: {count_above_18}")

count_below_15 = (df['pct_below_15'] >= 50).sum()
print(f"5. Buildings with >= 50% of area below 15ºC: {count_below_15}")
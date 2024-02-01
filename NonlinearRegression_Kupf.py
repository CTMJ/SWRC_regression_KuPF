from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# Read data from CSV file
df = pd.read_csv('./Datasets/KuPF_944_35CMdata.csv')

# Extract h and theta_v from the DataFrame
h = df['h_1281'].values
theta = df['theta_1281'].values

# Remove NaNs and infs (you can also use other methods to handle them)
valid_indices = ~(np.isnan(h) | np.isinf(h) | np.isnan(theta) | np.isinf(theta))
h = h[valid_indices]
theta = theta[valid_indices]

sorted_indices = np.argsort(h)
h_sorted = h[sorted_indices]
#print(h_sorted[-10:])
theta_sorted = theta[sorted_indices]
# print(theta_sorted[-10:])

#option a: Step weighting beyond a threshold
# Step weighting beyond a threshold
'''def step_weight(h, threshold, high_weight):
    return np.where(h > threshold, high_weight, 1)

# Parameters for step weighting - adjust threshold and high_weight as needed
threshold_drying = max(h_sorted) * 0.9  # 90% of the maximum h value
threshold_wetting = min(h_sorted) * 1.1  # 110% of the minimum h value
high_weight = 10

# Calculate weights
weights_drying = step_weight(h_sorted, threshold_drying, high_weight)
weights_wetting = step_weight(h_sorted, threshold_wetting, high_weight)

# Combine weights by taking the maximum at each point
sigma = np.maximum(weights_drying, weights_wetting)'''

# option b:Linear weighting towards specific h values
## Linear weighting towards specific h values
'''def linear_weight(h, most_weight_at_max):
    if most_weight_at_max:
        return (h - min(h)) / (max(h) - min(h))
    else:
        return (max(h) - h) / (max(h) - min(h))

# Calculate weights
weights_drying = linear_weight(h_sorted, most_weight_at_max=True)
weights_wetting = linear_weight(h_sorted, most_weight_at_max=False)

# Combine weights by taking the maximum at each point
sigma = np.maximum(weights_drying, weights_wetting)
'''
# option c: U-shaped weighting function
# U-shaped weighting function (inverted Gaussian)
def u_shaped_weight(h, center, width):
    return 1 - np.exp(-((h - center)**2) / (2 * width**2))

# Parameters for U-shaped weighting - adjust center and width as needed
center = np.mean(h_sorted)  # center of the h values
width = (max(h_sorted) - min(h_sorted)) / 2  # width of the distribution

# Calculate weights for each point using the U-shaped function
sigma = u_shaped_weight(h_sorted, center, width)

# Ensure that weights do not become zero or negative
sigma = np.maximum(sigma, 0.001)# replace 0.1 with the minimum weight you want to assign


# modifyed sigma

'''
theta_s = max(theta_sorted)  # or use an estimated value if you have one

# Calculate weights for the drying end
weights_drying = 1 / (0.1 + h_sorted)

# Calculate weights for the saturated end
weights_saturated = 1 / (0.1 + np.abs(theta_s - theta_sorted))

# Combine the weights by taking the maximum of the two weights at each point
sigma = np.minimum(weights_drying, weights_saturated)'''


print(list[sigma])

# Initial parameter bounds for the VGM model
param_bounds = ([0.3, 0.000, 1e-5, 1.0], [0.7, 0.3, 0.5, 17])

# VGM model function
def vgm_model(h, theta_s, theta_r, alpha, n):
    m = 1 - 1/n
    return theta_r + ((theta_s - theta_r) / ((1 + (alpha * h)**n)**m))

# Curve fitting with weights
popt, pcov = curve_fit(vgm_model, h_sorted, theta_sorted, bounds=param_bounds, sigma=sigma)

# popt, pcov = curve_fit(vgm_model, h_sorted, theta_sorted, sigma=sigma)

# Generate many points to create a smooth curve for the line plot
h_values_smooth = np.logspace(np.log10(h_sorted.min()), np.log10(h_sorted.max()*100), 1000)
theta_values_smooth = vgm_model(h_values_smooth, *popt)

# Extracted parameters
theta_s, theta_r, alpha, n = popt
print("Fitted Parameters:", popt)

# Plotting
plt.figure(figsize=(16, 6))
plt.scatter(h_sorted, theta_sorted, label='Original Data', color='red', s=50)  # s is the marker size
n = max(len(h_sorted) // 10, 1)  # Calculate n so that there are at most 10 annotations
for i, (h_val, theta_val) in enumerate(zip(h_sorted, theta_sorted)):
    if i % n == 0:  # Check if the index is divisible by n
        plt.annotate(f'{theta_val:.2f}', (h_val, theta_val),
                     textcoords="offset points", xytext=(0,10), ha='center')
plt.plot(h_values_smooth, theta_values_smooth, label='Fitted Curve', color='green', linewidth=2)
plt.xscale('symlog')  # Set x-axis to logarithmic scale
plt.xlabel('Pressure head (h)')
plt.ylabel('Soil water content (θ)')
plt.title('Water Retention curve')
custom_ticks = [1, 10, 100, 1000, 10000, 100000]
plt.xticks(custom_ticks, labels=[str(tick) for tick in custom_ticks])

plt.grid(True, which="both", ls="--")  # Grid lines and apply to both major and minor ticks
plt.legend()

# Calculate R^2 and RMSE for display
r2 = r2_score(theta_sorted, vgm_model(h_sorted, *popt))
rmse = np.sqrt(mean_squared_error(theta_sorted, vgm_model(h_sorted, *popt)))
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Add R^2 and RMSE to the plot
plt.text(0.1, 0.9, f'R^2: {r2:.2f}\nRMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=12)

plt.show()

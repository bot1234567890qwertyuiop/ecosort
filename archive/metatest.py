import matplotlib.pyplot as plt
import numpy as np

# Data
years = np.array([1921, 1931, 1941, 1951, 1961, 1971, 1981, 1991, 2001, 2011])
life_expectancy_a = np.array([59, 60, 63, 66, 68, 69, 72, 75, 77, 79])
life_expectancy_b = np.array([61, 62, 66, 71, 74, 76, 79, 81, 82, 84])

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Linear fits
ax1.scatter(years, life_expectancy_a, color='blue', label='Gender A')
ax1.scatter(years, life_expectancy_b, color='red', label='Gender B')

# Linear fits
coeffs_a_linear = np.polyfit(years, life_expectancy_a, 1)
coeffs_b_linear = np.polyfit(years, life_expectancy_b, 1)

# Create extended range for linear fits
years_extended = np.linspace(1920, 2025, 100)
fit_line_a = np.polyval(coeffs_a_linear, years_extended)
fit_line_b = np.polyval(coeffs_b_linear, years_extended)

ax1.plot(years_extended, fit_line_a, color='blue', linestyle='--', label='Gender A Linear')
ax1.plot(years_extended, fit_line_b, color='red', linestyle='--', label='Gender B Linear')

# Add arrows at the end of lines with increased size
ax1.annotate('', xy=(years_extended[-1], fit_line_a[-1]), 
            xytext=(years_extended[-5], fit_line_a[-5]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2, mutation_scale=20))
ax1.annotate('', xy=(years_extended[-1], fit_line_b[-1]), 
            xytext=(years_extended[-5], fit_line_b[-5]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, mutation_scale=20))

# Calculate and plot 2021 estimates for linear fit
year_2021 = 2021
estimate_2021_a_linear = np.polyval(coeffs_a_linear, year_2021)
estimate_2021_b_linear = np.polyval(coeffs_b_linear, year_2021)
ax1.scatter(year_2021, estimate_2021_a_linear, color='darkblue', marker='x', s=150, label='2021 Estimate A (Linear)')
ax1.scatter(year_2021, estimate_2021_b_linear, color='darkred', marker='x', s=150, label='2021 Estimate B (Linear)')

ax1.set_title("Linear Fits")
ax1.set_xlabel("Year")
ax1.set_ylabel("Life Expectancy")
ax1.legend()
ax1.grid(True)

# Plot 2: Polynomial fits
ax2.scatter(years, life_expectancy_a, color='blue', label='Gender A')
ax2.scatter(years, life_expectancy_b, color='red', label='Gender B')

# Polynomial fits (degree 2)
coeffs_a_poly = np.polyfit(years, life_expectancy_a, 2)
coeffs_b_poly = np.polyfit(years, life_expectancy_b, 2)

# Generate points for smooth curves with extended range
years_smooth = np.linspace(1920, 2025, 100)
fit_curve_a = np.polyval(coeffs_a_poly, years_smooth)
fit_curve_b = np.polyval(coeffs_b_poly, years_smooth)

ax2.plot(years_smooth, fit_curve_a, color='blue', label='Gender A Polynomial')
ax2.plot(years_smooth, fit_curve_b, color='red', label='Gender B Polynomial')

# Add arrows at the end of curves with increased size
ax2.annotate('', xy=(years_smooth[-1], fit_curve_a[-1]), 
            xytext=(years_smooth[-5], fit_curve_a[-5]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2, mutation_scale=20))
ax2.annotate('', xy=(years_smooth[-1], fit_curve_b[-1]), 
            xytext=(years_smooth[-5], fit_curve_b[-5]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, mutation_scale=20))

# Calculate and plot 2021 estimates for polynomial fit
estimate_2021_a_poly = np.polyval(coeffs_a_poly, year_2021)
estimate_2021_b_poly = np.polyval(coeffs_b_poly, year_2021)
ax2.scatter(year_2021, estimate_2021_a_poly, color='darkblue', marker='x', s=150, label='2021 Estimate A (Polynomial)')
ax2.scatter(year_2021, estimate_2021_b_poly, color='darkred', marker='x', s=150, label='2021 Estimate B (Polynomial)')

ax2.set_title("Polynomial Fits (Degree 2)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Life Expectancy")
ax2.legend()
ax2.grid(True)

# Print the estimates
print(f"Linear fit estimates for 2021:")
print(f"Gender A: {estimate_2021_a_linear:.1f} years")
print(f"Gender B: {estimate_2021_b_linear:.1f} years")
print(f"\nPolynomial fit estimates for 2021:")
print(f"Gender A: {estimate_2021_a_poly:.1f} years")
print(f"Gender B: {estimate_2021_b_poly:.1f} years")

# Adjust layout and display
plt.tight_layout()
plt.show()

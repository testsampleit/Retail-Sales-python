# ============================================
# STATISTICS 101 PROJECT
# Retail Sales & Customer Behavior Analysis
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, binom, poisson, ttest_1samp
from sklearn.linear_model import LinearRegression

# --------------------------------------------
# 1. DATA CREATION (Population & Sample)
# --------------------------------------------

np.random.seed(1)
n = 250  # number of days

df = pd.DataFrame({
    "Day": range(1, n + 1),
    "Customers": np.random.poisson(200, n),
    "Ad_Spend": np.random.normal(1000, 300, n).clip(200),
    "Discount_Percent": np.random.choice([0, 5, 10, 15], n),
})

df["Daily_Sales"] = (
    20 * df["Customers"]
    + 0.05 * df["Ad_Spend"]
    + 30 * df["Discount_Percent"]
    + np.random.normal(0, 500, n)
).clip(0)

print(df.head())

# --------------------------------------------
# 2. DATA ORGANIZATION & FREQUENCY DISTRIBUTION
# --------------------------------------------

print("\nDiscount Frequency:")
print(df["Discount_Percent"].value_counts())

# --------------------------------------------
# 3. DATA VISUALIZATION
# --------------------------------------------

plt.hist(df["Daily_Sales"], bins=20)
plt.title("Histogram of Daily Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(df["Daily_Sales"])
plt.title("Boxplot of Daily Sales")
plt.show()

plt.scatter(df["Customers"], df["Daily_Sales"])
plt.title("Customers vs Daily Sales")
plt.xlabel("Customers")
plt.ylabel("Sales")
plt.show()

# --------------------------------------------
# 4. MEASURES OF CENTRAL TENDENCY
# --------------------------------------------

mean_sales = df["Daily_Sales"].mean()
median_sales = df["Daily_Sales"].median()
mode_sales = df["Daily_Sales"].round(-2).mode()[0]

print("\nCentral Tendency:")
print("Mean:", mean_sales)
print("Median:", median_sales)
print("Mode:", mode_sales)

# --------------------------------------------
# 5. MEASURES OF DISPERSION
# --------------------------------------------

range_sales = df["Daily_Sales"].max() - df["Daily_Sales"].min()
variance = df["Daily_Sales"].var()
std_dev = df["Daily_Sales"].std()
iqr = df["Daily_Sales"].quantile(0.75) - df["Daily_Sales"].quantile(0.25)
cv = (std_dev / mean_sales) * 100

print("\nDispersion:")
print("Range:", range_sales)
print("Variance:", variance)
print("Std Dev:", std_dev)
print("IQR:", iqr)
print("Coefficient of Variation:", cv)

# --------------------------------------------
# 6. SKEWNESS & KURTOSIS
# --------------------------------------------

print("\nSkewness:", skew(df["Daily_Sales"]))
print("Kurtosis:", kurtosis(df["Daily_Sales"]))

# --------------------------------------------
# 7. CORRELATION ANALYSIS
# --------------------------------------------

print("\nCorrelation Matrix:")
print(df[["Customers", "Ad_Spend", "Discount_Percent", "Daily_Sales"]].corr())

# --------------------------------------------
# 8. LINEAR REGRESSION
# --------------------------------------------

X = df[["Customers", "Ad_Spend", "Discount_Percent"]]
y = df["Daily_Sales"]

model = LinearRegression()
model.fit(X, y)

print("\nRegression Coefficients:")
print("Customers:", model.coef_[0])
print("Ad Spend:", model.coef_[1])
print("Discount:", model.coef_[2])
print("Intercept:", model.intercept_)

# --------------------------------------------
# 9. PROBABILITY BASICS
# --------------------------------------------

prob_high_sales = len(df[df["Daily_Sales"] > 10000]) / len(df)
print("\nProbability (Sales > 10,000):", prob_high_sales)

# --------------------------------------------
# 10. PROBABILITY DISTRIBUTIONS
# --------------------------------------------

x = np.linspace(df["Daily_Sales"].min(), df["Daily_Sales"].max(), 200)
pdf = norm.pdf(x, mean_sales, std_dev)

plt.plot(x, pdf)
plt.title("Normal Distribution of Daily Sales")
plt.show()

# Binomial Distribution
binom_prob = binom.pmf(3, 7, prob_high_sales)
print("Binomial Probability:", binom_prob)

# Poisson Distribution (customers per hour)
poisson_prob = poisson.pmf(5, 4)
print("Poisson Probability:", poisson_prob)

# --------------------------------------------
# 11. SAMPLING & CENTRAL LIMIT THEOREM
# --------------------------------------------

sample_means = []
for _ in range(1000):
    sample_means.append(df["Daily_Sales"].sample(40).mean())

plt.hist(sample_means, bins=25)
plt.title("Sampling Distribution of Mean Sales")
plt.show()

# --------------------------------------------
# 12. HYPOTHESIS TESTING
# --------------------------------------------

# H0: Mean Daily Sales = 8000
t_stat, p_val = ttest_1samp(df["Daily_Sales"], 8000)

print("\nHypothesis Test:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("Reject H0: Mean sales differs from 8000")
else:
    print("Fail to Reject H0")

# --------------------------------------------
# END OF PROJECT
# --------------------------------------------

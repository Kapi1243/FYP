import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import prepare_data

data = prepare_data(save_scaler=False)
df = data["df"]

# -------------------------------
# 0. Setup
# -------------------------------
os.makedirs("Figures", exist_ok=True)

# -------------------------------
# 1. Target Distribution
# -------------------------------
ax = df["target"].value_counts().plot(kind="bar", color=["steelblue", "salmon"], edgecolor="black")
ax.set_xticklabels(["No Disease (0)", "Disease (1)"], rotation=0)
plt.title("Heart Disease Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Figures/eda_target_distribution.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 2. Feature Histograms
# -------------------------------
df[["age", "chol", "thalach", "trestbps", "oldpeak"]].hist(
    bins=20, figsize=(12, 8), color="steelblue", edgecolor="black"
)
plt.suptitle("Distribution of Key Continuous Features", y=1.02)
plt.tight_layout()
plt.savefig("Figures/eda_feature_histograms.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 3. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(12, 9))
sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("Figures/eda_correlation_matrix.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 4. Boxplot — Age by Target
# -------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(x="target", y="age", data=df, palette=["steelblue", "salmon"])
plt.xticks([0, 1], ["No Disease", "Disease"])
plt.title("Age Distribution by Heart Disease Status (Boxplot)")
plt.xlabel("Heart Disease Status")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("Figures/eda_boxplot_age.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 5. Violinplot — Age by Target
# -------------------------------
plt.figure(figsize=(7, 5))
sns.violinplot(x="target", y="age", data=df, palette=["steelblue", "salmon"])
plt.xticks([0, 1], ["No Disease", "Disease"])
plt.title("Age Distribution by Heart Disease Status (Violin)")
plt.xlabel("Heart Disease Status")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("Figures/eda_violinplot_age.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 6. Violinplot — Max Heart Rate by Target
# -------------------------------
plt.figure(figsize=(7, 5))
sns.violinplot(x="target", y="thalach", data=df, palette=["steelblue", "salmon"])
plt.xticks([0, 1], ["No Disease", "Disease"])
plt.title("Max Heart Rate Distribution by Heart Disease Status")
plt.xlabel("Heart Disease Status")
plt.ylabel("Max Heart Rate (thalach)")
plt.tight_layout()
plt.savefig("Figures/eda_violinplot_thalach.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 7. Pairplot — Key Clinical Features
# -------------------------------
pairplot_df = df[["age", "chol", "thalach", "oldpeak", "target"]].copy()
pairplot_df["target"] = pairplot_df["target"].map({0: "No Disease", 1: "Disease"})

sns.pairplot(pairplot_df, hue="target", palette={"No Disease": "steelblue", "Disease": "salmon"})
plt.suptitle("Pairplot of Key Clinical Features", y=1.02)
plt.savefig("Figures/eda_pairplot.png", bbox_inches="tight")
plt.show()

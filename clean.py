import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load your dataset
df = pd.read_csv("handwriting_personality_large_dataset.csv")  # Replace with your filename

# List of traits to analyze
traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']  # Example traits, modify as needed

# Handwriting features
feature_cols = [f'Feature_{i}' for i in range(1, 16)]

# Store results
results = []

# Loop through each trait
for trait in traits:
    # Loop through each feature and calculate correlation
    for feature in feature_cols:
        # Drop rows with missing data in relevant columns
        subset = df[[trait, feature]].dropna()

        # Calculate Pearson correlation
        r, p_value = pearsonr(subset[trait], subset[feature])
        results.append({
            'Trait': trait,
            'Feature': feature,
            'Correlation': r,
            'P-value': p_value,
            'Significant': p_value < 0.05
        })

# Convert to DataFrame and sort by absolute correlation
results_df = pd.DataFrame(results).sort_values(by='Correlation', key=abs, ascending=False)

# Print results
print(results_df)

# Optionally, save the results to a new CSV file
results_df.to_csv('results.csv', index=False)

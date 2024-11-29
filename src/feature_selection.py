# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from preprocessing import data,file_path
# %%
data.head()


# %%
# Function to calculate WoE for a given feature
def calculate_woe(data, feature, Grade):
    """
    Calculate Weight of Evidence (WoE) for a categorical feature.
    
    Parameters:
        data (pd.DataFrame): The dataset
        feature (str): The feature/column name for which to calculate WoE
        target (str): The target/column name
    
    Returns:
        pd.DataFrame: A DataFrame with WoE values for each category
    """
    # Create a table of counts for each category in the feature and the target
    crosstab = pd.crosstab(data[feature], data[Grade], normalize='columns')
    crosstab = crosstab.rename(columns={0: 'Non-Event', 1: 'Event'})
    
    # Add WoE column
    crosstab['WoE'] = np.log(crosstab['Event'] / crosstab['Non-Event'])
    return crosstab[['WoE']]

# Initialize a dictionary to store WoE for all features
woe_dict = {}

# Iterate over each column except the target
target_column = 'Grade'  # Update if the target column has a different name
for column in data.columns:
    if column != target_column:
        try:
            woe_dict[column] = calculate_woe(data, column, target_column)
            print(f"WoE calculated for feature: {column}")
        except Exception as e:
            print(f"Could not calculate WoE for feature: {column}. Error: {e}")

# Display WoE for all features
for feature, woe_values in woe_dict.items():
    print(f"\nFeature: {feature}")
    print(woe_values)
    print(len(woe_values))


# %%
data = pd.read_csv(file_path)

# Function to calculate WoE for a given feature
def calculate_woe(data, feature, Grade):
    """
    Calculate Weight of Evidence (WoE) for a categorical feature.
    
    Parameters:
        data (pd.DataFrame): The dataset
        feature (str): The feature/column name for which to calculate WoE
        target (str): The target/column name
    
    Returns:
        pd.Series: A Series with WoE values for each category
    """
    # Create a table of counts for each category in the feature and the target
    crosstab = pd.crosstab(data[feature], data[Grade], normalize='columns')
    crosstab = crosstab.rename(columns={0: 'Non-Event', 1: 'Event'})
    
    # Add WoE column
    crosstab['WoE'] = np.log(crosstab['Event'] / crosstab['Non-Event'])
    return crosstab['WoE']

# Initialize a dictionary to store WoE for all features
woe_dict = {}

# Iterate over each column except the target
target_column = 'Grade'  # Update if the target column has a different name
for column in data.columns:
    if column != target_column:
        try:
            woe_values = calculate_woe(data, column, target_column)
            # Store the maximum WoE value for ranking
            woe_dict[column] = woe_values.max()  # Use max to get the most influential WoE
            print(f"WoE calculated for feature: {column}")
        except Exception as e:
            print(f"Could not calculate WoE for feature: {column}. Error: {e}")

# Convert the dictionary to a DataFrame for better visualization
woe_df = pd.DataFrame.from_dict(woe_dict, orient='index', columns=['Max WoE'])

# Rank the features based on Max WoE
woe_df['Rank'] = woe_df['Max WoE'].rank(ascending=False)

# Sort the DataFrame by Rank
woe_df = woe_df.sort_values(by='Rank')

# Display the ranked features
print("\nRanked Features based on Weight of Evidence (WoE):")
print(woe_df)

# %%


# Drop rows with missing values
data_cleaned = data.dropna()

# Define the target variable and features
target_column = 'Grade'  # Update if the target column has a different name
X = data_cleaned.drop(columns=[target_column])  # Features
y = data_cleaned[target_column]  # Target variable

# Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression()

# Initialize RFE
rfe = RFE(estimator=model)

# Create a parameter grid for the number of features to select
param_grid = {'n_features_to_select': np.arange(1, X_train.shape[1] + 1)}

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=rfe, param_grid=param_grid, scoring='accuracy', cv=5)

# Fit the GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get the best number of features
best_n_features = grid_search.best_params_['n_features_to_select']
best_score = grid_search.best_score_

# Display the results
print(f"Best number of features to select: {best_n_features}")
print(f"Best cross-validated accuracy: {best_score}")

# Optionally, you can refit RFE with the best number of features
rfe_best = RFE(estimator=model, n_features_to_select=best_n_features)
rfe_best.fit(X_train_scaled, y_train)

# Get the ranking of features
ranking_best = rfe_best.ranking_

# Create a DataFrame to show the features and their ranking
feature_ranking_best_df = pd.DataFrame({'Feature': X.columns, 'Rank': ranking_best})

# Sort the DataFrame by Rank
feature_ranking_best_df = feature_ranking_best_df.sort_values(by='Rank')

# Display the ranked features based on the best number of features
print("\nRanked Features based on the best number of features selected:")
print(feature_ranking_best_df)

# Optionally, you can also see which features were selected
selected_features_best = feature_ranking_best_df[feature_ranking_best_df['Rank'] == 1]
print("\nSelected Features based on the best number of features:")
print(selected_features_best)

# %%
# Define the target variable and features
target_column = 'Grade'  # Update if the target column has a different name
X = data_cleaned.drop(columns=[target_column])  # Features
y = data_cleaned[target_column]  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to determine the number of components that explain 95% of the variance
pca = PCA(n_components=None)  # Keep all components initially
X_pca = pca.fit_transform(X_scaled)

# Calculate the cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components needed to explain 95% of the variance
n_components_95 = np.argmax(explained_variance >= 0.95) + 1
print(f"Number of components to explain 95% variance: {n_components_95}")

# Apply PCA again with the optimal number of components
pca_final = PCA(n_components=n_components_95)
X_pca_final = pca_final.fit_transform(X_scaled)

# Get the components and explained variance for analysis
pca_components = pca_final.components_
explained_variance_final = pca_final.explained_variance_ratio_

# Display the explained variance of each component
print("\nExplained variance ratio for each component:")
for i, ev in enumerate(explained_variance_final):
    print(f"Component {i+1}: {ev:.4f}")

# Create a DataFrame showing feature contributions to each principal component
pca_feature_contributions = pd.DataFrame(
    pca_components.T, 
    index=X.columns, 
    columns=[f"PC{i+1}" for i in range(n_components_95)]
)

# Calculate the absolute contributions
pca_feature_contributions_abs = pca_feature_contributions.abs()

# Sum the absolute contributions across all principal components
total_contributions = pca_feature_contributions_abs.sum(axis=1)

# Create a DataFrame for total contributions and sort it
total_contributions_df = pd.DataFrame(total_contributions, columns=['Total Contribution'])
total_contributions_df = total_contributions_df.sort_values(by='Total Contribution', ascending=False)

# Add a rank column to the DataFrame
total_contributions_df['Rank'] = total_contributions_df['Total Contribution'].rank(ascending=False, method='min')

# Reorder the DataFrame to show rank first
total_contributions_df = total_contributions_df.reset_index().rename(columns={'index': 'Feature'})
total_contributions_df = total_contributions_df[['Rank', 'Feature', 'Total Contribution']]

# Display the ranked features based on total contributions
print("\nRanking of features based on total contributions to selected principal components:")
print(total_contributions_df.sort_values(by='Rank'))

# %%
# Rankings from PCA
pca_rankings = pd.DataFrame({
    'Feature': ['SMARCA4', 'BCOR', 'RB1', 'GRIN2A', 'MUC16', 'PIK3R1', 'PDGFRA', 'EGFR', 
                'NF1', 'CSMD3', 'NOTCH1', 'IDH2', 'Gender', 'PIK3CA', 'PTEN', 
                'Race', 'FAT4', 'FUBP1', 'Age_at_diagnosis', 'TP53', 'CIC', 'ATRX', 'IDH1'],
    'Total Contribution': [3.903822, 3.841967, 3.837974, 3.776082, 3.720309, 3.709714, 
                          3.701858, 3.678714, 3.674851, 3.522081, 3.498505, 3.390264, 
                          3.322463, 3.304892, 3.303130, 3.126706, 3.098197, 2.948243, 
                          2.288513, 2.236392, 2.153581, 2.056945, 1.701671]
})

# Rankings from RFE
rfe_rankings = pd.DataFrame({
    'Feature': ['Age_at_diagnosis', 'IDH1', 'ATRX', 'TP53', 'PTEN', 'NOTCH1', 
                'PIK3R1', 'NF1', 'IDH2', 'GRIN2A'],
    'Rank': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
})

# Rankings from WoE
woe_rankings = pd.DataFrame({
    'Feature': ['Age_at_diagnosis', 'RB1', 'PTEN', 'IDH1', 'GRIN2A', 'PDGFRA', 
                'EGFR', 'Race', 'PIK3R1', 'MUC16', 'NF1', 'CSMD3', 'FAT4', 
                'ATRX', 'TP53', 'CIC', 'PIK3CA', 'Gender', 'FUBP1', 'NOTCH1', 
                'IDH2', 'SMARCA4', 'BCOR'],
    'WoE': [float('inf'), 2.059234, 1.859347, 1.457252, 1.374455, 1.305462, 
            1.285095, 0.917697, 0.776618, 0.654112, 0.594923, 0.547776, 
            0.411644, 0.369657, 0.271667, 0.236664, 0.187432, 0.102411,0.086742, 
            0.081241, 0.038380, 0.036951, 0.000846]
})

# Check lengths
print(f"PCA Features Length: {len(pca_rankings)}")
print(f"RFE Features Length: {len(rfe_rankings)}")
print(f"WoE Features Length: {len(woe_rankings)}")

# Select top features from each method
top_n = 10  # Define how many top features to select
top_pca_features = pca_rankings.head(top_n)['Feature'].tolist()
top_rfe_features = rfe_rankings['Feature'].tolist()
top_woe_features = woe_rankings.head(top_n)['Feature'].tolist()

# Combine all selected features into a single set
combined_features = set(top_pca_features + top_rfe_features + top_woe_features)

# Create a DataFrame for combined features and their counts
combined_rankings = pd.DataFrame(combined_features, columns=['Feature'])

# Count occurrences in each ranking
combined_rankings['PCA Count'] = combined_rankings['Feature'].apply(lambda x: x in top_pca_features)
combined_rankings['RFE Count'] = combined_rankings['Feature'].apply(lambda x: x in top_rfe_features)
combined_rankings['WoE Count'] = combined_rankings['Feature'].apply(lambda x: x in top_woe_features)

# Calculate total count of appearances in top rankings
combined_rankings['Total Count'] = combined_rankings[['PCA Count', 'RFE Count', 'WoE Count']].sum(axis=1)

# Sort the combined rankings by total count
best_features = combined_rankings.sort_values(by='Total Count', ascending=False)

# Display the best features
print("Best Features Selected Based on PCA, RFE, and WoE Rankings:")
print(best_features[['Feature', 'Total Count']])

# %%
# DataFrames for rankings from WoE, RFE, and PCA
woe_ranking = pd.DataFrame({
    "Feature": ["Age_at_diagnosis", "RB1", "PTEN", "IDH1", "GRIN2A", "PDGFRA", "EGFR", "Race", "PIK3R1", "MUC16",
                "NF1", "CSMD3", "FAT4", "ATRX", "TP53", "CIC", "PIK3CA", "Gender", "FUBP1", "NOTCH1", "IDH2", 
                "SMARCA4", "BCOR"],
    "Rank": [1, 2.059234, 1.859347, 1.457252, 1.374455, 1.305462, 1.285095, 0.917697, 0.776618, 0.654112, 
             0.594923, 0.547776, 0.411644, 0.369657, 0.271667, 0.236664, 0.187432, 0.102411, 0.086742, 
             0.081241, 0.038380, 0.036951, 0.000846]
})

rfe_selected_features = ["Age_at_diagnosis", "IDH1", "ATRX", "TP53", "PTEN", "NOTCH1", "PIK3R1", "NF1", "IDH2", "GRIN2A"]

pca_ranking = pd.DataFrame({
    "Feature": ["SMARCA4", "BCOR", "RB1", "GRIN2A", "MUC16", "PIK3R1", "PDGFRA", "EGFR", "NF1", "CSMD3", 
                "NOTCH1", "IDH2", "Gender", "PIK3CA", "PTEN", "Race", "FAT4", "FUBP1", "Age_at_diagnosis", 
                "TP53", "CIC", "ATRX", "IDH1"],
    "Rank": [3.903822, 3.841967, 3.837974, 3.776082, 3.720309, 3.709714, 3.701858, 3.678714, 3.674851, 
             3.522081, 3.498505, 3.390264, 3.322463, 3.304892, 3.303130, 3.126706, 3.098197, 2.948243, 
             2.288513, 2.236392, 2.153581, 2.056945, 1.701671]
})

# Normalize rankings (scaling between 0 and 1)
def normalize_ranking(df, rank_col):
    df[rank_col] = (df[rank_col] - df[rank_col].min()) / (df[rank_col].max() - df[rank_col].min())
    return df

woe_ranking = normalize_ranking(woe_ranking, "Rank")
pca_ranking = normalize_ranking(pca_ranking, "Rank")

# Merge rankings
combined_ranking = woe_ranking.merge(pca_ranking, on="Feature", suffixes=("_woe", "_pca"), how="outer")

# Add RFE selection (binary: 1 if selected, 0 otherwise)
combined_ranking["RFE"] = combined_ranking["Feature"].apply(lambda x: 1 if x in rfe_selected_features else 0)

# Compute overall score (average of normalized ranks and RFE contribution)
combined_ranking["Combined_Score"] = combined_ranking[["Rank_woe", "Rank_pca", "RFE"]].mean(axis=1)

# Select the best features based on combined score
best_features = combined_ranking.sort_values("Combined_Score", ascending=False).head(10)

print("Best Features:")
print(best_features)


# %%
# Select top 15 features
top_15_features = combined_ranking.head(15)

# Print the ranked features
print("All Ranked Features:")
print(combined_ranking)

print("\nTop 15 Features:")
print(top_15_features)


# %%
combined_ranking['Feature']

# %%
combined_ranking['Feature']

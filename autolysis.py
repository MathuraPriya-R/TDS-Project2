# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "requests",
#   "scikit-learn",
#   "scipy",
#   "geopandas",
#   "networkx",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
import geopandas as gpd
import networkx as nx

def main(csv_file):
    
    # Validate input
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} does not exist.")
        sys.exit(1)

    # Load data
    # data = pd.read_csv(csv_file)
    data = read_csv_with_fallback(csv_file)
    print("Data loaded successfully.")

    # Automatically determine category based on CSV filename (or custom logic)
    category = determine_category(csv_file)
    if not category:
        print("Could not determine category from the file. Exiting.")
        sys.exit(1)

    # Create necessary directories for the category
    create_category_directories(category)

    # Perform analysis
    analysis_results = analyze_data(data)

    # Generate visualizations
    generate_visualizations(data, category)

    # Narrate the story
    narrate_story(csv_file, analysis_results, category)

def read_csv_with_fallback(filepath):
    # List of encodings to try
    encodings = ['utf-8', 'ISO-8859-1', 'utf-16', 'utf-8-sig', 'latin1', 'cp1252']
    
    # Try reading the file with different encodings
    for encoding in encodings:
        try:
            print(f"Trying to read the file with encoding: {encoding}")
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"File successfully read with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to read with encoding: {encoding}")
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    raise ValueError("Could not read the file with any of the tried encodings.")

def determine_category(csv_file):
    """Automatically determine the category from the file name or content."""
    if "goodreads" in csv_file.lower():
        return "goodreads"
    elif "happiness" in csv_file.lower():
        return "happiness"
    elif "media" in csv_file.lower():
        return "media"
    else:
        return None

def create_category_directories(category):
    """Create the category folder structure."""
    base_path = os.path.join(os.getcwd(), category)
    os.makedirs(base_path, exist_ok=True)
    # os.makedirs(os.path.join(base_path, "images"), exist_ok=True)  # Subfolder for images
    print(f"Category directories created for: {category}")

def analyze_data(data):
    """Perform advanced analysis on the dataset."""
    summary = {
        "Shape": data.shape,
        "Columns": data.dtypes.to_dict(),
        "Missing Values": data.isnull().sum().to_dict(),
        "Descriptive Stats": data.describe().to_dict(),
    }

    # 1. Outlier and Anomaly Detection (Z-Score)
    outliers = detect_outliers(data)
    summary["Outliers"] = outliers

    # 2. Correlation Analysis
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        correlation = numeric_data.corr()
        summary["Correlation"] = correlation.to_dict()

    # 3. Regression Analysis (Simple Linear Regression Example)
    regression_results = regression_analysis(data)
    summary["Regression"] = regression_results

    # 4. Feature Importance (Random Forest Example)
    feature_results = feature_importance(data)
    summary["Feature Importance"] = feature_results

    # 5. Time Series Analysis (if applicable)
    time_series_result = time_series_analysis(data)
    summary["Time Series Analysis"] = time_series_result

    # 6. Cluster Analysis
    cluster_results = cluster_analysis(data)
    summary["Cluster Analysis"] = cluster_results

    # 7. Geographic Analysis (if applicable)
    geographic_result = geographic_analysis(data)
    summary["Geographic Analysis"] = geographic_result

    # 8. Network Analysis (Placeholder for actual network analysis)
    network_results = network_analysis(data)
    summary["Network Analysis"] = network_results

    return summary

def detect_outliers(data):
    """Detect outliers using Z-score."""
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    outliers = (z_scores > 3).sum(axis=0)
    return outliers

def regression_analysis(data):
    """Perform simple linear regression (predict a numeric column)."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return "Not enough numeric columns for regression."
    
    X = data[numeric_cols[:-1]]
    y = data[numeric_cols[-1]]
    
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    X = StandardScaler().fit_transform(X)
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    return {"Coefficients": model.coef_, "Intercept": model.intercept_, "MSE": mse}

def feature_importance(data):
    """Perform feature importance using RandomForest."""
    numeric_cols = data.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] < 2:
        return "Not enough numeric columns for feature importance."
    
    X = numeric_cols.iloc[:, :-1]
    y = numeric_cols.iloc[:, -1]
    
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    importance = model.feature_importances_
    feature_names = numeric_cols.columns[:-1]
    return dict(zip(feature_names, importance))

def time_series_analysis(data):
    """Perform time series analysis (if applicable)."""
    if 'Date' not in data.columns:
        return "No 'Date' column found for time series analysis."
    
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    data.set_index('Date', inplace=True)
    
    monthly_data = data.resample('M').mean()
    return monthly_data.describe().to_dict()

def cluster_analysis(data):
    """Perform cluster analysis using KMeans."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return "No numeric columns for clustering."
    
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(numeric_data)
    return {"Cluster Centers": kmeans.cluster_centers_, "Labels": clusters.tolist()}

def geographic_analysis(data):
    """Perform geographic analysis (if applicable)."""
    if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
        return "No geographic data found for analysis."
    
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['Longitude'], data['Latitude']))
    return gdf.geometry.describe().to_dict()

def network_analysis(data):
    """
    Placeholder for network analysis.
    Assuming 'Source' and 'Target' columns in data to represent connections.
    """
    if 'Source' not in data.columns or 'Target' not in data.columns:
        return "No 'Source' and 'Target' columns found for network analysis."
    
    G = nx.from_pandas_edgelist(data, source='Source', target='Target')
    
    network_summary = {
        "Number of Nodes": len(G.nodes),
        "Number of Edges": len(G.edges),
        "Average Degree": sum(dict(G.degree()).values()) / len(G.nodes) if G.nodes else 0
    }
    
    return network_summary

def generate_visualizations(data, category):
    """Generate and save visualizations based on the dataset and category."""
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        print("No numeric data available for correlation analysis.")
        return

    correlation = numeric_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    
    output_dir = os.path.join(category)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

def narrate_story(csv_file, analysis_results, category):
    """Use LLM to create a story and save it as README.md."""
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("AIPROXY_TOKEN")
        }
    
    prompt = (
        f"Analyze the dataset {csv_file}. Here are the key findings:\n"
        f"{analysis_results}\n"
        "Write a story narrating the analysis, insights, and implications."
    )
    # print("Prompt: ", prompt)

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt}
            ]
        })
    )
    
    if response.status_code == 200:
        print("Story generated successfully.")
        story = response.json()["choices"][0]["message"]["content"]
        
        readme_path = os.path.join(category, "README.md")
        os.makedirs(os.path.dirname(readme_path), exist_ok=True)
        with open(readme_path, "w") as f:
            f.write(story)
    else:
        print(f"Error generating story: {response.status_code}, {response.text}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    main(sys.argv[1])

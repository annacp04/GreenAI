import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from zero_shot_pollution import AirQualityAnalyzer

# EXAMPLE EXECUTION: 
analyzer = AirQualityAnalyzer()
test_images = ["polluted.jpg", "not_polluted.png"]

for img_path in test_images:
    try:
        print(f"\n{'='*20} ANALYZING: {img_path} {'='*20}")
        results = analyzer.analyze_image(img_path)

        print("\n[1] Classification Probabilities:")
        for label, prob in results['probs'].items():
            print(f"  - {label}: {prob:.4f}")

        print("\n[2] Intensity Scores (Cosine Similarity):")
        print(f"  - Pollution Intensity: {results['polluted_score']:.4f}")
        print(f"  - Clean Sky Intensity: {results['clean_score']:.4f}")
        print(f"  - Net Balance:         {results['balance']:.4f} (Positive = Polluted)")
        
        print(f"\n[3] Feature Vector Size: {results['image_vector'].shape}")
        
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")

def prepare_features(image_vector, temp, humidity, clip_balance):
    """
    Unifies all inputs into a single feature vector for the model.
    """

    # Decide on the standarizatin and ponderations...

    # 1. Flatten the 512 CLIP vector
    img_features = image_vector.flatten()
    # 2. Sensor array (Raw values are fine for Random Forest)
    sensors = np.array([temp, humidity, clip_balance])
    # 3. Final Concatenation (Total dimensions: 515)
    return np.concatenate([img_features, sensors])


# --- DATA COLLECTION SIMULATION ---
X = [] 
y = [] 

# IMPORTANT: You need to fill these lists with your actual data
# Example of how you would add one data point:
# feat = prepare_features(your_clip_vector, 22.5, 60, 0.12)
# X.append(feat)
# y.append(40.5) 

# Ensure we have data before continuing
if len(X) > 1:
    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluation
    predictions = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Model RMSE: {rmse:.2f} µg/m3 of NO2")

    # Feature Importance Analysis
    importances = rf_model.feature_importances_
    print("\n--- Feature Importance ---")
    print(f"Temperature:     {importances[-3]:.4f}")
    print(f"Humidity:        {importances[-2]:.4f}")
    print(f"CLIP Vis Score:  {importances[-1]:.4f}")
    print(f"Image Features:  {np.sum(importances[:-3]):.4f}")
else:
    print("Error: Not enough data points to train the model.")
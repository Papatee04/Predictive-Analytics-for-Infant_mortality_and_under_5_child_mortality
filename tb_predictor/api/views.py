# api/views.py
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # For SMOTE
from sklearn.utils import class_weight  # For cost-sensitive learning

# Load DHS dataset without any conversions (ensure the file path is correct)
file_path = 'C:/dhs data/datasets/ZMIR71DT/ZMIR71FL.DTA'
df = pd.read_stata(file_path, convert_categoricals=False)

# Define selected features
selected_features = [
    'v106',  # Mother’s education
    'v190',  # Wealth index
    'v201',  # Total children ever born
    'v012',  # Mother’s age
    'v208',  # Births in last five years
    'v025',  # Place of residence (urban/rural)
    'v127',  # Main floor material
    'v113',  # Source of drinking water
    'v116',  # Type of toilet facility
    'v119'   # Household has electricity
]

# Target variable
target_variable = 'b5_01'  # First child is alive

# Preprocess the data and train the models


def train_models():
    # Filter out rows where the target variable is missing
    df_clean = df.dropna(subset=[target_variable] + selected_features).copy()

    # Convert all relevant columns to numeric
    df_clean[selected_features + [target_variable]] = df_clean[selected_features +
                                                               [target_variable]].apply(pd.to_numeric, errors='coerce')

    # Split data into features (X) and target (y)
    X = df_clean[selected_features]
    y = df_clean[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Resampling the minority class using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Decision Tree Model with Cost-sensitive Learning
    class_weights_dt = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    decision_tree_model = DecisionTreeClassifier(random_state=42, class_weight={
                                                 0: class_weights_dt[0], 1: class_weights_dt[1]})
    decision_tree_model.fit(X_train_resampled, y_train_resampled)

    # Random Forest Model with Cost-sensitive Learning
    class_weights_rf = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    random_forest_model = RandomForestClassifier(random_state=42, class_weight={
                                                 0: class_weights_rf[0], 1: class_weights_rf[1]})
    random_forest_model.fit(X_train_resampled, y_train_resampled)

    # SVM Model with Cost-sensitive Learning
    svm_model = SVC(class_weight='balanced', random_state=42)
    svm_model.fit(X_train_resampled, y_train_resampled)

    return decision_tree_model, random_forest_model, svm_model


# Train models
decision_tree_model, random_forest_model, svm_model = train_models()

# Define a function to preprocess input data


def preprocess_input(data):
    # Convert the scalar input data into a list of dictionaries
    if isinstance(data, dict):
        data = [data]  # Wrap it into a list to treat it as a single row
    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


@api_view(['POST'])
def predict(request):
    if request.method == 'POST':
        # Get data from request
        input_data = request.data

        # Preprocess the input data
        df_input = preprocess_input(input_data)

        # Ensure the input is formatted correctly
        if df_input.shape[1] != len(selected_features):
            return Response({"error": "Input data must have the correct number of features."}, status=status.HTTP_400_BAD_REQUEST)

        # Make predictions with the Decision Tree model as an example
        dt_predictions = decision_tree_model.predict(df_input)
        rf_predictions = random_forest_model.predict(df_input)
        svm_predictions = svm_model.predict(df_input)

        return Response({
            "decision_tree_predictions": dt_predictions.tolist(),
            "random_forest_predictions": rf_predictions.tolist(),
            "svm_predictions": svm_predictions.tolist()
        }, status=status.HTTP_200_OK)

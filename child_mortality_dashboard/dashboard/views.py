import joblib
from django.shortcuts import render, redirect
from .forms import ChildMortalityFormForm
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.transform import dodge
from bokeh.models import ColumnDataSource
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from dashboard.models import *
import numpy as np
import lime
import lime.lime_tabular
import json
import html
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Load the pre-trained model
model = joblib.load(
    'C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/random_forest_model.pkl')

def generate_lime_explanation(model, feature_names, X_input):
    """
    Generate LIME explanation for a single prediction
    
    Parameters:
    - model: Trained machine learning model
    - feature_names: List of feature names used in the model
    - X_input: Input features for a single prediction
    
    Returns:
    - HTML-formatted LIME explanation
    """
    # Load the training data to help LIME understand feature distributions
    # Assuming you have a training dataset saved
    X_train = joblib.load('C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/X_train.pkl')
    
    # Convert input to NumPy array
    X_input = np.array(X_input).reshape(1, -1)
    
    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['High Mortality Risk', 'Low Mortality Risk'],
        verbose=True,
        mode='classification'
    )
    
    # Generate explanation
    explanation = explainer.explain_instance(
        X_input[0], 
        model.predict_proba, 
        num_features=10,  # Number of top features to explain
        top_labels=1
    )
    
    # Convert explanation to HTML
    explanation_html = "<h3>Feature Importance for Prediction</h3>"
    explanation_html += "<table class='w-full border-collapse'>"
    explanation_html += "<tr><th class='border p-2'>Feature</th><th class='border p-2'>Impact</th></tr>"
    
    for feature, impact in explanation.as_list(label=explanation.top_labels[0]):
        # Escape HTML to prevent XSS
        safe_feature = html.escape(feature)
        
        # Color-code impact (green for positive, red for negative)
        color_class = "text-green-600" if impact > 0 else "text-red-600"
        explanation_html += (
            f"<tr>"
            f"<td class='border p-2'>{safe_feature}</td>"
            f"<td class='border p-2 {color_class}'>{impact:.4f}</td>"
            f"</tr>"
        )
    
    explanation_html += "</table>"
    
    # Use:
    proba = model.predict_proba(np.array(X_input).reshape(1, -1))[0]
    # Use:
    prediction = model.predict(np.array(X_input).reshape(1, -1))[0]
    
    prediction_text = "Low Mortality Risk" if prediction == 1 else "High Mortality Risk"
    prediction_color = "text-green-600" if prediction == 1 else "text-red-600"
    
    explanation_html += (
        f"<div class='mt-4'>"
        f"<p>Prediction: <span class='{prediction_color} font-bold'>{prediction_text}</span></p>"
        f"<p>Probability of Low Risk: {proba[1]:.2%}</p>"
        f"<p>Probability of High Risk: {proba[0]:.2%}</p>"
        f"</div>"
    )
    
    return explanation_html

def create_3d_risk_scatter_plot(data):
    """
    Create a 3D scatter plot for multidimensional risk analysis
    
    Parameters:
    - data: DataFrame containing risk assessment features
    
    Returns:
    - Plotly figure object for 3D scatter plot
    """
    # Ensure you have the necessary columns
    required_columns = [
        'number_of_children', 
        'age_at_first_sex', 
        'total_children_ever_born',
        'risk_prediction'
    ]
    
    # Select relevant features for visualization
    X = data[required_columns[:-1]]
    y = data['risk_prediction']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optional: Use PCA for dimensionality reduction if needed
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create color mapping for risk levels
    colors = ['red' if pred == 0 else 'green' for pred in y]
    
    # Create 3D scatter plot
    trace = go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            opacity=0.7,
            colorscale='Viridis'
        ),
        text=[f"Risk Level: {'High' if pred == 0 else 'Low'}" for pred in y],
        hoverinfo='text'
    )
    
    layout = go.Layout(
        scene=dict(
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            zaxis_title='Third Principal Component',
            aspectmode='cube'
        ),
        title='Multidimensional Risk Analysis Scatter Plot',
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    
    # Convert to JSON for frontend rendering
    plot_json = fig.to_json()
    
    return plot_json

def dashboard(request):
    # Get actual counts from the database
    total_assessments = ChildMortalityAssessment.objects.count()
    high_risk_cases = ChildMortalityAssessment.objects.filter(risk_prediction=1).count()
    active_monitoring = ChildMortalityAssessment.objects.filter(status='active').count()
    
    # Calculate success rate (as in the previous version)
    total_high_risk = ChildMortalityAssessment.objects.filter(risk_prediction=1).count()
    successful_interventions = ChildMortalityAssessment.objects.filter(
        risk_prediction=1, 
        intervention_outcome='successful'
    ).count()
    
    success_rate = (successful_interventions / total_high_risk * 100) if total_high_risk > 0 else 0

    all_assessments = ChildMortalityAssessment.objects.all()
    assessments_df = pd.DataFrame.from_records(all_assessments.values())
    
    # Generate 3D scatter plot
    risk_scatter_plot = create_3d_risk_scatter_plot(assessments_df)

    # Model data for visualization
    model_data = [
        {'metric': 'Accuracy', 'Logistic Regression': 0.999269, 'Random Forest': 1},
        {'metric': 'Precision', 'Logistic Regression': 0.9990234, 'Random Forest': 1},
        {'metric': 'Recall', 'Logistic Regression': 1, 'Random Forest': 1},
        {'metric': 'AUC', 'Logistic Regression': 0.9985528, 'Random Forest': 1},
    ]

    # Extract data for Bokeh plotting
    metrics = [d['metric'] for d in model_data]
    logistic_regression = [d['Logistic Regression'] for d in model_data]
    random_forest = [d['Random Forest'] for d in model_data]

    # Create a Bokeh figure
    plot = figure(x_range=metrics, title="Model Comparison",
                  toolbar_location="above", tools="pan,wheel_zoom,box_zoom,reset")

    # Convert data to ColumnDataSource for interactivity and efficient data handling
    source = ColumnDataSource(data=dict(
        metrics=metrics,
        logistic_regression=logistic_regression,
        random_forest=random_forest,
    ))

    # Add bars for Logistic Regression and Random Forest with an offset using dodge
    plot.vbar(x=dodge('metrics', -0.15, range=plot.x_range), top='logistic_regression', width=0.3, source=source, legend_label="Logistic Regression", color="#8884d8")
    plot.vbar(x=dodge('metrics', 0.15, range=plot.x_range), top='random_forest', width=0.3, source=source, legend_label="Random Forest", color="#82ca9d")

    # Customize plot appearance
    plot.xgrid.grid_line_color = None
    plot.y_range.start = 0
    plot.legend.title = "Models"
    plot.legend.label_text_font_size = "10pt"

    # Generate script and div components for embedding
    script, div = components(plot)

    # Key insights to display
    key_insights = [
        "Maternal health and obstetric care are critical factors in child mortality.",
        "Family size and resource strain impact child survival rates.",
        "Marital status and social support affect child mortality risks.",
        "Maternal education and family planning are associated with lower child mortality.",
    ]

    if request.method == 'POST':
        form = ChildMortalityFormForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # Define feature names to match your model's input
            feature_names = [
                'cause_of_fistula',
                'number_of_children',
                'current_pregnancy',
                'number_of_living_children',
                'marital_status',
                'entries_in_birth_history',
                'total_children_ever_born',
                'age_at_first_sex',
                'ever_been_married'
            ]

            # Prepare input for prediction and LIME
            X_input = [
                int(data['cause_of_fistula']),
                data['number_of_children'],
                int(data['current_pregnancy']),
                data['number_of_living_children'],
                int(data['marital_status']),
                int(data['entries_in_birth_history']),
                int(data['total_children_ever_born']),
                int(data['age_at_first_sex']),
                int(data['ever_been_married'])
            ]

            # Make prediction
            # Use:
            prediction = model.predict(np.array(X_input).reshape(1, -1))[0]

            # Generate LIME explanation
            lime_explanation = generate_lime_explanation(model, feature_names, X_input)

            # Create a new ChildMortalityAssessment object
            new_assessment = ChildMortalityAssessment.objects.create(
                form_data=None,  # Temporarily bypass the requirement
                user=request.user, 
                risk_prediction=prediction,
                status='active',  # Set initial status
                intervention_outcome='pending'  # Set initial intervention outcome
                # You might want to add more fields from the form data
            )

            # Map prediction result to user-friendly message
            if prediction == 1:
                prediction_message = "Low chance of child mortality"
            else:
                prediction_message = "High chance of child mortality"

            # Recalculate counts after new assessment
            total_assessments = ChildMortalityAssessment.objects.count()
            high_risk_cases = ChildMortalityAssessment.objects.filter(risk_prediction=1).count()
            active_monitoring = ChildMortalityAssessment.objects.filter(status='active').count()

            # Render with updated information
            return render(request, 'dashboard/dashboard.html', {
                'form': form,
                'prediction_message': prediction_message,
                'script': script,
                'div': div,
                'key_insights': key_insights,
                'total_assessments': total_assessments,
                'high_risk_cases': high_risk_cases,
                'active_monitoring': active_monitoring,
                'success_rate': success_rate,
                'lime_explanation': lime_explanation
            })

    else:
        form = ChildMortalityFormForm()

    return render(request, 'dashboard/dashboard.html', {
        'form': form,
        'script': script,
        'div': div,
        'key_insights': key_insights,
        'total_assessments': total_assessments,
        'high_risk_cases': high_risk_cases,
        'active_monitoring': active_monitoring,
        'success_rate': success_rate,
        'lime_explanation': None
    })


def success(request):
    prediction = request.GET.get('prediction')
    return render(request, 'dashboard/success.html', {'prediction': prediction})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'dashboard/login.html', {'form': form})

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'dashboard/signup.html', {'form': form})

@login_required
def profile_view(request):
    return render(request, 'dashboard/profile.html')

def logout_view(request):
    logout(request)
    return redirect('dashboard')
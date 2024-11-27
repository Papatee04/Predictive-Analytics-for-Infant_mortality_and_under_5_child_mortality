import joblib
from django.shortcuts import render, redirect
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.transform import dodge
from bokeh.models import ColumnDataSource, HoverTool
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dashboard.forms import *
import io
import base64


# Load pre-trained models
u5mr_model = joblib.load('C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/rf_classifier_balanced_u5mr.pkl')
imr_model = joblib.load('C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/rf_classifier_balanced_imr.pkl')

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
    - data: Django QuerySet of ChildMortalityAssessment or DataFrame
    
    Returns:
    - Plotly figure JSON for 3D scatter plot
    """
    # If input is a DataFrame, process directly
    if isinstance(data, pd.DataFrame):
        assessment_data = data
    else:
        # Convert QuerySet to DataFrame
        assessment_data = pd.DataFrame(list(data.values()))
    
    # Required columns for the plot
    required_columns = ['number_of_children', 'age_at_first_sex', 'total_children_ever_born', 'risk_prediction']
    
    # Check if the required columns exist in the DataFrame
    missing_columns = [col for col in required_columns if col not in assessment_data.columns]
    
    if missing_columns:
        # If columns are missing, try to retrieve data from related form_data
        try:
            # Fetch related ChildMortalityForm data
            form_data = ChildMortalityForm.objects.all().values(
                'number_of_children', 
                'age_at_first_sex', 
                'total_children_ever_born'
            )
            form_df = pd.DataFrame(list(form_data))
            
            # Merge form data with assessment data
            assessment_data = pd.merge(
                assessment_data, 
                form_df, 
                left_index=True, 
                right_index=True
            )
        except Exception as e:
            # If merging fails, return an empty plot
            print(f"Could not retrieve additional data: {e}")
            return go.Figure().to_json()
    
    # Validate column existence after potential merge
    missing_columns = [col for col in required_columns if col not in assessment_data.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return go.Figure().to_json()
    
    # Prepare data for PCA
    X = assessment_data[['number_of_children', 'age_at_first_sex', 'total_children_ever_born']]
    y = assessment_data['risk_prediction']
    
    # Convert categorical columns to numeric if needed
    # This is necessary if the values are stored as strings or categorical
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop rows with NaN values
    X_clean = X.dropna()
    y_clean = y[X_clean.index]
    
    # Check if we have enough data
    if len(X_clean) < 2:
        return go.Figure().to_json()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Use PCA for dimensionality reduction
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create color mapping for risk levels
    colors = ['red' if pred == 0 else 'green' for pred in y_clean]
    
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
        text=[f"Risk Level: {'High' if pred == 0 else 'Low'}" for pred in y_clean],
        hoverinfo='text'
    )
    
    # Variance explained by each component
    variance_explained = pca.explained_variance_ratio_
    
    layout = go.Layout(
        scene=dict(
            xaxis_title=f'PC1 ({variance_explained[0]*100:.2f}%)',
            yaxis_title=f'PC2 ({variance_explained[1]*100:.2f}%)',
            zaxis_title=f'PC3 ({variance_explained[2]*100:.2f}%)',
            aspectmode='cube'
        ),
        title='Multidimensional Risk Analysis Scatter Plot',
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    
    # Convert to JSON for frontend rendering
    return fig.to_json()

def dashboard(request):
    # Database statistics
    total_assessments = ChildMortalityAssessment.objects.count()
    high_risk_cases = ChildMortalityAssessment.objects.filter(risk_prediction=1).count()
    active_monitoring = ChildMortalityAssessment.objects.filter(status='active').count()
    
    # Calculate success rate
    total_high_risk = ChildMortalityAssessment.objects.filter(risk_prediction=1).count()
    successful_interventions = ChildMortalityAssessment.objects.filter(
        risk_prediction=1, 
        intervention_outcome='successful'
    ).count()
    
    success_rate = (successful_interventions / total_high_risk * 100) if total_high_risk > 0 else 0

    # Feature importance data (example, replace with actual data)
    feature_importance_data = {
        'Feature': ['b12', 'v101', 'v113', 'socioeconomic_index', 'm70', 
                    'v106', 'wealth_residence_interaction', 'v190', 
                    'health_access_index', 'v025'],
        'Importance': [0.45, 0.35, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02]
    }

    # Create Bokeh feature importance plot
    script, div = create_feature_importance_plot(feature_importance_data)


    # Key insights
    key_insights = [
        "Maternal health and socioeconomic factors critically impact child mortality.",
        "Education and health access significantly influence child survival rates.",
        "Wealth index and residence type play crucial roles in child health outcomes.",
        "Postnatal care is a key determinant of infant and child mortality.",
    ]

    # Initialize context with default values
    context = {
        'form': None,
        'key_insights': key_insights,
        'total_assessments': total_assessments,
        'high_risk_cases': high_risk_cases,
        'active_monitoring': active_monitoring,
        'success_rate': success_rate,
        'has_prediction': False,
        'script': script,
        'div': div,
        'lime_explanation': None
    }

    if request.method == 'POST':
        form = MortalityPredictionForm(request.POST)
        context['form'] = form

        if form.is_valid():
            # Extract form data
            data = {
                'v106': form.cleaned_data['education_level'],
                'v025': form.cleaned_data['residence_type'],
                'v190': form.cleaned_data['wealth_index'],
                'v101': form.cleaned_data['region'],
                'v113': form.cleaned_data['water_source'],
                'm70': form.cleaned_data['postnatal_check'],
                'b12': form.cleaned_data.get('succeeding_birth_interval', 0),
                'socioeconomic_index': calculate_socioeconomic_index(
                    form.cleaned_data['wealth_index'], 
                    form.cleaned_data['education_level'], 
                    form.cleaned_data['residence_type']
                ),
                'health_access_index': calculate_health_access_index(
                    form.cleaned_data['water_source'], 
                    form.cleaned_data['postnatal_check']
                ),
                'wealth_residence_interaction': int(form.cleaned_data['wealth_index']) * int(form.cleaned_data['residence_type'])
            }
    
            # Convert to DataFrame
            input_data = pd.DataFrame([data])

            # Check which button was pressed
            calculate_type = request.POST.get('calculate')
            if calculate_type == 'u5mr':
                prediction = u5mr_model.predict(input_data)[0]
                probability = u5mr_model.predict_proba(input_data)[0][1]
                result_label = "Under-Five Mortality Risk"
                
                # Add LIME explanation for U5MR
                context = add_lime_explanation_to_context(
                    request, 
                    context, 
                    input_data, 
                    'u5mr'
                )
                
            elif calculate_type == 'imr':
                prediction = imr_model.predict(input_data)[0]
                probability = imr_model.predict_proba(input_data)[0][1]
                result_label = "Infant Mortality Risk"
                
                # Add LIME explanation for IMR
                context = add_lime_explanation_to_context(
                    request, 
                    context, 
                    input_data, 
                    'imr'
                )

            # Map prediction to a user-friendly message
            prediction_label = get_prediction_label(prediction)    

            # Create assessment record
            assessment = ChildMortalityAssessment.objects.create(
                user=request.user,
                risk_prediction=prediction
            )

            # Update context with prediction details
            context.update({
                'assessment': assessment,
                'prediction': prediction_label,
                'probability': probability,
                'result_label': result_label,
                'has_prediction': True
            })

            return render(request, 'dashboard/dashboard.html', context)
    else:
        context['form'] = MortalityPredictionForm()

    return render(request, 'dashboard/dashboard.html', context)

def generate_lime_explanation(model, input_data, training_data_path, feature_names=None):
    """
    Generate a LIME explanation for a given model and input data.
    
    Args:
        model: Trained machine learning model
        input_data (pd.DataFrame): Single row of input data to explain
        training_data_path (str): Path to the pickle file with training data
        feature_names (list, optional): List of feature names 
    
    Returns:
        dict: Containing HTML-formatted explanation and visualization
    """
    # Load training data
    training_data = joblib.load(training_data_path)
    X_train = training_data['X_train']
    
    # If feature names not provided, use columns of input data
    if feature_names is None:
        feature_names = input_data.columns.tolist()
    
    # Prepare data for LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, 
        feature_names=feature_names, 
        class_names=['Low Risk', 'High Risk'],
        discretize_continuous=True
    )
    
    # Convert input data to numpy array for LIME
    input_array = input_data.values[0]
    
    # Generate explanation
    explanation = explainer.explain_instance(
        input_array, 
        model.predict_proba, 
        num_features=10
    )
    
    # Create HTML explanation
    html_explanation = "<div class='lime-explanation'>"
    html_explanation += "<h3>Local Interpretable Model-Agnostic Explanations (LIME)</h3>"
    
    # Add text explanation
    html_explanation += "<div class='explanation-text'>"
    html_explanation += "<p><strong>Prediction Probability:</strong> {:.2f}%</p>".format(
        model.predict_proba(input_data)[0][1] * 100
    )
    html_explanation += "<p><strong>Key Influencing Factors:</strong></p>"
    html_explanation += "<ul>"
    
    # Sort features by absolute importance
    sorted_features = sorted(
        explanation.as_list(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    for feature, importance in sorted_features:
        direction = "positive" if importance > 0 else "negative"
        html_explanation += f"<li>{feature}: <span class='text-{direction}'>{importance:.4f}</span></li>"
    
    html_explanation += "</ul>"
    html_explanation += "</div>"
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    explanation.as_pyplot_figure()
    plt.title("Feature Importance in Prediction")
    plt.tight_layout()
    
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Add visualization to HTML
    html_explanation += f"""
    <div class='explanation-visualization'>
        <img src="data:image/png;base64,{image_base64}" alt="LIME Explanation Visualization" 
             style="max-width:100%; height:auto;"/>
    </div>
    """
    
    html_explanation += "</div>"
    
    return {
        'html': html_explanation,
        'probability': model.predict_proba(input_data)[0][1]
    }

def add_lime_explanation_to_context(request, context, input_data, model_type):
    """
    Add LIME explanation to context based on model type
    
    Args:
        request: Django request object
        context: Current context dictionary
        input_data: Prepared input data for prediction
        model_type: 'u5mr' or 'imr'
    
    Returns:
        Updated context with LIME explanation
    """
    # Choose appropriate model and training data path
    if model_type == 'u5mr':
        model = u5mr_model
        training_data_path = 'C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/u5mr_training_data.pkl'
    elif model_type == 'imr':
        model = imr_model
        training_data_path = 'C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/imr_training_data.pkl'
    else:
        return context
    
    # Feature names matching your form data
    feature_names = [
        'Education Level', 'Residence Type', 'Wealth Index', 'Region', 
        'Water Source', 'Postnatal Check', 'Birth Interval', 
        'Socioeconomic Index', 'Health Access Index', 
        'Wealth-Residence Interaction'
    ]
    
    # Generate LIME explanation
    try:
        lime_result = generate_lime_explanation(
            model, 
            input_data, 
            training_data_path, 
            feature_names
        )
        
        context['lime_explanation'] = lime_result['html']
        context['lime_probability'] = lime_result['probability']
    except Exception as e:
        context['lime_explanation'] = f"Error generating LIME explanation: {str(e)}"
    
    return context

def create_feature_importance_plot(feature_importance_data):
    # Prepare data
    source = ColumnDataSource(data=feature_importance_data)

    # Create the figure with a wide range of interactive tools
    plot = figure(
        y_range=feature_importance_data['Feature'], 
        title="Feature Importance from Random Forest (Balanced Data)",
        toolbar_location="above",
        tools="pan,box_zoom,zoom_in,zoom_out,reset,save,wheel_zoom"
    )
    
    # Add horizontal bar chart
    plot.hbar(
        y='Feature', 
        right='Importance', 
        height=0.4, 
        source=source, 
        color='skyblue'
    )

    # Add hover tool to show importance values
    hover = HoverTool(tooltips=[("Feature", "@Feature"), ("Importance", "@Importance")])
    plot.add_tools(hover)

    # Customize the plot's appearance
    plot.xaxis.axis_label = "Importance"
    plot.yaxis.axis_label = "Feature"
    plot.yaxis.major_label_text_font_size = "10pt"
    plot.title.text_font_size = "14pt"
    plot.outline_line_color = None  # Hide the plot outline for a cleaner look
    
    return components(plot)

def calculate_socioeconomic_index(wealth_index, education_level, residence_type):
    """Calculate composite socioeconomic index"""
    return (int(wealth_index) * 0.4) + (int(education_level) * 0.3) + (int(residence_type) == 1) * 0.3

def calculate_health_access_index(water_source, postnatal_check):
    """Calculate health access index"""
    return (int(water_source) in [1, 2]) * 0.5 + (postnatal_check) * 0.5

# Function to map prediction values to user-friendly labels
def get_prediction_label(prediction):
    return "Low Risk of Mortality , classification: 0" if prediction == 0 else "High Risk of Mortality , classification: 1"


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
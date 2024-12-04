import joblib
from django.shortcuts import get_object_or_404, render, redirect
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
from lime import lime_tabular
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

def dashboard(request):
    # Database statistics
    total_assessments = ChildMortalityAssessment.objects.count()
    high_risk_cases = ChildMortalityAssessment.objects.filter(risk_prediction=1).count()
    active_monitoring = ChildMortalityAssessment.objects.filter(status='active').count()

    # Calculate success rate
    successful_interventions = ChildMortalityAssessment.objects.filter(
        risk_prediction=1,
        intervention_outcome='successful'
    ).count()

    success_rate = (successful_interventions / high_risk_cases * 100) if high_risk_cases > 0 else 0

    # Feature importance data
    feature_importance_data = {
        'Feature': ['b12-Succeeding birth interval (months)', 'v101-Region', 'v113-Source of drinking water', 'socioeconomic_index', 'm70-Baby postnatal check within 2 months ', 
                    'v106-Highest educational level', 'wealth_residence_interaction', 'v190-Wealth index combined', 
                    'health_access_index', 'v025-Type of place of residence '],
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

    # Initialize context
    context = {
        'form': MortalityPredictionForm(),
        'key_insights': key_insights,
        'total_assessments': total_assessments,
        'high_risk_cases': high_risk_cases,
        'active_monitoring': active_monitoring,
        'success_rate': success_rate,
        'has_prediction': False,
        'script': script,
        'div': div,
        'lime_explanation': None,
        # Add mortality rate metrics
        'neonatal_mortality_rate': 27.32,
        'postneonatal_mortality_rate': 28.82,
        'infant_mortality_rate': 56.14,
        'child_mortality_rate': 33.98,
        'under_five_mortality_rate': 88.22,
        'vaccinations': ['BCG', 'Hepatitis B', 'Rotavirus', 'DPT', 'Polio', 'Pneumococcal']
    }

    # Handle POST request
    if request.method == 'POST':
        form = MortalityPredictionForm(request.POST)
        context['form'] = form

        if form.is_valid():
            try:
                # Extract and process form data
                input_data = prepare_input_data(form.cleaned_data)
                calculate_type = request.POST.get('calculate')
                
                # Perform prediction
                if calculate_type == 'u5mr':
                    context = handle_prediction(request, context, input_data, 'u5mr', u5mr_model)
                elif calculate_type == 'imr':
                    context = handle_prediction(request, context, input_data, 'imr', imr_model)

                return render(request, 'dashboard/dashboard.html', context)
            except Exception as e:
                context['error'] = f"An error occurred during prediction: {str(e)}"

    return render(request, 'dashboard/dashboard.html', context)

def prepare_input_data(cleaned_data):
    return pd.DataFrame([{
        'v106': int(cleaned_data['education_level']),
        'v025': int(cleaned_data['residence_type']),
        'v190': int(cleaned_data['wealth_index']),
        'v101': int(cleaned_data['region']),
        'v113': int(cleaned_data['water_source']),
        'm70': int(cleaned_data['postnatal_check']),
        'b12': int(cleaned_data.get('succeeding_birth_interval', 0)),
        'socioeconomic_index': calculate_socioeconomic_index(
            cleaned_data['wealth_index'], 
            cleaned_data['education_level'], 
            cleaned_data['residence_type']
        ),
        'health_access_index': calculate_health_access_index(
            cleaned_data['water_source'], 
            cleaned_data['postnatal_check']
        ),
        'wealth_residence_interaction': int(cleaned_data['wealth_index']) * int(cleaned_data['residence_type'])
    }])
    
def handle_prediction(request, context, input_data, model_type, model):
    """
    Handle prediction logic and update context.
    """
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        result_label = "Under-Five Mortality Risk" if model_type == 'u5mr' else "Infant Mortality Risk"
        prediction_label = get_prediction_label(prediction)
        
        # Add LIME explanation
        context = add_lime_explanation_to_context(request, context, input_data, model_type)

        # Create assessment record
        assessment = ChildMortalityAssessment.objects.create(
            user=request.user,
            risk_prediction=prediction
        )

        context.update({
            'assessment': assessment,
            'prediction': prediction_label,
            'probability': probability,
            'result_label': result_label,
            'has_prediction': True
        })
    except Exception as e:
        context['error'] = f"Error during {model_type} prediction: {str(e)}"
    
    return context

def generate_lime_explanation(model, input_data, training_data_path, feature_names=None):
    """
    Generate a LIME explanation for a given model and input data.
    """
    # Load training data
    training_data = joblib.load(training_data_path)
    X_train = training_data['X_train']

    # Set feature names if not provided
    if feature_names is None:
        feature_names = input_data.columns.tolist()

    # Prepare LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['Low Risk', 'High Risk'],
        discretize_continuous=True
    )

    # Generate explanation
    input_array = input_data.values[0]
    explanation_low = explainer.explain_instance(
        input_array,
        model.predict_proba,
        labels=[0],  # Low Risk
        num_features=10
    )
    explanation_high = explainer.explain_instance(
        input_array,
        model.predict_proba,
        labels=[1],  # High Risk
        num_features=10
    )

    # Combine explanations in HTML
    html_explanation = f"<h3>Low Risk Explanation</h3>{explanation_low.as_html()}"
    html_explanation += f"<h3>High Risk Explanation</h3>{explanation_high.as_html()}"

    return {
        'html': html_explanation,
        'probability': model.predict_proba(input_data)[0][1]
    }

def format_lime_explanation_html(explanation, model, input_data):
    """
    Format LIME explanation into HTML.
    """
    html_explanation = "<div class='lime-explanation'>"
    html_explanation += "<h3>LIME Explanation</h3>"

    # Add text explanation
    html_explanation += "<ul>"
    for feature, importance in sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True):
        direction = "positive" if importance > 0 else "negative"
        html_explanation += f"<li>{feature}: <span class='text-{direction}'>{importance:.4f}</span></li>"
    html_explanation += "</ul>"

    # Generate plot
    plt.figure(figsize=(10, 6))
    explanation.as_pyplot_figure()
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    html_explanation += f'<img src="data:image/png;base64,{image_base64}" alt="Explanation Visualization"/>'
    html_explanation += "</div>"

    return html_explanation

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

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

@login_required
def profile_edit_view(request):
    if request.method == 'POST':
        user = request.user
        
        # Update username if changed and not empty
        new_username = request.POST.get('username', '').strip()
        if new_username and new_username != user.username:
            # Check if username is already taken
            from django.contrib.auth.models import User
            if User.objects.filter(username=new_username).exists():
                messages.error(request, 'Username is already taken.')
            else:
                user.username = new_username
        
        # Update first and last name
        user.first_name = request.POST.get('first_name', '').strip()
        user.last_name = request.POST.get('last_name', '').strip()
        
        try:
            user.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile_edit')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
    
    return render(request, 'dashboard/profile.html')

def logout_view(request):
    logout(request)
    return redirect('dashboard')

@login_required
def assessments_management(request):
    # Basic statistics
    total_assessments = ChildMortalityAssessment.objects.count()
    high_risk_cases = ChildMortalityAssessment.objects.filter(risk_prediction=1).count()
    active_monitoring = ChildMortalityAssessment.objects.filter(status='active').count()

    # Success rate calculation
    successful_interventions = ChildMortalityAssessment.objects.filter(
        risk_prediction=1, 
        intervention_outcome='successful'
    ).count()
    success_rate = (successful_interventions / high_risk_cases * 100) if high_risk_cases > 0 else 0

    # Get all assessments for the table
    assessments = ChildMortalityAssessment.objects.filter(user=request.user).order_by('-assessment_date')

    context = {
        'total_assessments': total_assessments,
        'high_risk_cases': high_risk_cases,
        'active_monitoring': active_monitoring,
        'success_rate': success_rate,
        'assessments': assessments,
        'form': MortalityPredictionForm(),
    }

    return render(request, 'dashboard/assessments_management.html', context)

@login_required
def update_assessment(request, assessment_id):
    assessment = get_object_or_404(ChildMortalityAssessment, id=assessment_id, user=request.user)
    
    if request.method == 'POST':
        assessment.status = request.POST.get('status')
        assessment.intervention_outcome = request.POST.get('intervention_outcome')
        assessment.save()
    
    return redirect('assessments_management')

@login_required
def delete_assessment(request, assessment_id):
    assessment = get_object_or_404(ChildMortalityAssessment, id=assessment_id, user=request.user)
    
    if request.method == 'POST':
        assessment.delete()
    
    return redirect('assessments_management')

@login_required
def assessment_details(request, assessment_id):
    assessment = get_object_or_404(ChildMortalityAssessment, id=assessment_id, user=request.user)
    
    context = {
        'assessment': assessment,
    }
    
    return render(request, 'dashboard/assessment_details.html', context)

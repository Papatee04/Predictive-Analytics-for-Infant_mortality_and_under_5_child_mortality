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

# Load the pre-trained model
model = joblib.load(
    'C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/random_forest_model.pkl')

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

            # Ensure to capture all the required fields
            X_input = [
                int(data['cause_of_fistula']),      # s1205
                data['number_of_children'],         # v218
                int(data['current_pregnancy']),     # v220
                # v219 (Add this to the form)
                data['number_of_living_children'],  # 219
                int(data['marital_status']),        # v502
                # v224 (Add this to the form)
                int(data['entries_in_birth_history']),
                # v201 (Add this to the form)
                int(data['total_children_ever_born']),
                int(data['age_at_first_sex']),      # v525
                # v535 (Add this to the form)
                int(data['ever_been_married'])
            ]

            prediction = model.predict([X_input])[0]

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
                'success_rate': success_rate
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
        'success_rate': success_rate
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
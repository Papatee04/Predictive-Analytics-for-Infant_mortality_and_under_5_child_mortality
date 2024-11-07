import joblib
from django.shortcuts import render
from .forms import ChildMortalityFormForm
from bokeh.plotting import figure
from bokeh.embed import components

# Load the pre-trained model
model = joblib.load(
    'C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/random_forest_model.pkl')

def dashboard(request):
    prediction = None

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
              toolbar_location=None, tools="")

    # Add bars for Logistic Regression and Random Forest
    plot.vbar(x=metrics, top=logistic_regression, width=0.3, legend_label="Logistic Regression", color="#8884d8")
    plot.vbar(x=metrics, top=random_forest, width=0.3, legend_label="Random Forest", color="#82ca9d")

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

            # Map prediction result to user-friendly message
            if prediction == 1:
                prediction_message = "Low chance of child mortality"
            else:
                prediction_message = "High chance of child mortality"

            return render(request, 'dashboard/dashboard.html', {
                'form': form,
                'prediction_message': prediction_message,
                'script': script,
                'div': div,
                'key_insights': key_insights
            })
        else:
            print("Form errors:", form.errors)
    else:
        form = ChildMortalityFormForm()

    return render(request, 'dashboard/dashboard.html', {
        'form': form,
        'prediction': prediction,
        'script': script,
        'div': div,
        'key_insights': key_insights
    })


def success(request):
    prediction = request.GET.get('prediction')
    return render(request, 'dashboard/success.html', {'prediction': prediction})

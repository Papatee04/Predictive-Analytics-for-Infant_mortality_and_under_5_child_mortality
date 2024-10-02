import joblib
from django.shortcuts import render, redirect
from .forms import ChildMortalityFormForm

# Load the pre-trained model
model = joblib.load(
    'C:/Users/tlche/OneDrive/Documents/GitHub/Predictive-Analytics-for-Tuberculosis-TB-Incidence-and-Treatment-Adherence/random_forest_model.pkl')


def dashboard(request):
    prediction = None
    if request.method == 'POST':
        form = ChildMortalityFormForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # Ensure to capture all the required fields
            X_input = [
                int(data['cause_of_fistula']),
                data['number_of_children'],  # Make sure this is correctly used
                int(data['current_pregnancy']),
                int(data['marital_status']),
                int(data['age_at_first_sex']),
                data['entries_in_birth_history'],  # New
                data['total_children_ever_born'],  # New
                int(data['ever_married_or_in_union']),  # New
                # You may need to adjust indices depending on your model
            ]

            prediction = model.predict([X_input])[0]

            return render(request, 'dashboard/dashboard.html', {'form': form, 'prediction': prediction})
        else:
            print("Form errors:", form.errors)
    else:
        form = ChildMortalityFormForm()

    return render(request, 'dashboard/dashboard.html', {'form': form, 'prediction': prediction})


def success(request):
    prediction = request.GET.get('prediction')
    return render(request, 'dashboard/success.html', {'prediction': prediction})

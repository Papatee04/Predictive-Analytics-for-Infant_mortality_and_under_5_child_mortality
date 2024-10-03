from django import forms
from .models import ChildMortalityForm


class ChildMortalityFormForm(forms.ModelForm):
    CAUSE_OF_FISTULA_CHOICES = [
        (1, 'Sexual assault'),
        (2, 'Pelvic surgery'),
        (6, 'Other'),
        (8, 'Don\'t know'),
    ]

    CURRENT_PREGNANCY_CHOICES = [
        (0, 'No current pregnancy'),
        (1, '1 child'),
        (2, '2 children'),
        (3, '3 children'),
        (4, '4 children'),
        (5, '5 children'),
        (6, '6+ children'),
    ]

    MARITAL_STATUS_CHOICES = [
        (0, 'Never in union'),
        (1, 'Currently in union/living with a man'),
        (2, 'Formerly in union/living with a man'),
    ]

    AGE_AT_FIRST_SEX_CHOICES = [(i, f'{i} years') for i in range(7, 50)] + [
        (0, 'Not had sex'),
        (96, 'At first union'),
        (98, 'Don\'t know'),
    ]

    USE_FAMILY_PLANNING_CHOICES = [
        (0, 'No'),
        (1, 'Yes'),
    ]

    cause_of_fistula = forms.ChoiceField(
        choices=CAUSE_OF_FISTULA_CHOICES, label="Cause of Fistula"
    )
    number_of_children = forms.IntegerField(
        min_value=0, max_value=20, label="Number of Living Children"
    )
    number_of_living_children = forms.IntegerField(
        min_value=0, max_value=20, label="Number of Living Children + Current Pregnancy"
    )

    current_pregnancy = forms.ChoiceField(
        choices=CURRENT_PREGNANCY_CHOICES, label="Living Children + Current Pregnancy"
    )
    marital_status = forms.ChoiceField(
        choices=MARITAL_STATUS_CHOICES, label="Marital Status"
    )
    age_at_first_sex = forms.ChoiceField(
        choices=AGE_AT_FIRST_SEX_CHOICES, label="Age at First Sex"
    )
    use_family_planning = forms.ChoiceField(
        choices=USE_FAMILY_PLANNING_CHOICES, label="Use of Family Planning"
    )
    education_years = forms.IntegerField(
        min_value=0, max_value=25, label="Number of Years of Education"
    )
    entries_in_birth_history = forms.IntegerField(
        min_value=0, max_value=20, label="Entries in Birth History"
    )
    total_children_ever_born = forms.IntegerField(
        min_value=0, max_value=20, label="Total Children Ever Born"
    )
    ever_been_married = forms.ChoiceField(
        choices=[(0, 'No'), (1, 'Formerly married'), (2, 'Lived with a man')],
        label="Ever Been Married or in Union"
    )

    class Meta:
        model = ChildMortalityForm
        fields = [
            'cause_of_fistula',
            'number_of_children',
            'current_pregnancy',
            'number_of_living_children',  # Add this line
            'marital_status',
            'age_at_first_sex',
            'entries_in_birth_history',
            'total_children_ever_born',
            'ever_been_married',
        ]

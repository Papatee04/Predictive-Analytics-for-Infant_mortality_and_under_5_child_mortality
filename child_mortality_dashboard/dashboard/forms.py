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

    cause_of_fistula = forms.ChoiceField(
        choices=CAUSE_OF_FISTULA_CHOICES, label="Cause of Fistula"
    )

    number_of_children = forms.IntegerField(
        min_value=0, max_value=20, label="Number of Living Children"
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

    class Meta:
        model = ChildMortalityForm
        fields = [
            'cause_of_fistula',
            'number_of_children',
            'current_pregnancy',
            'marital_status',
            'age_at_first_sex',
            'use_family_planning',
            'education_years',
        ]

    def clean_current_pregnancy(self):
        current_pregnancy = int(self.cleaned_data.get('current_pregnancy'))
        if current_pregnancy not in dict(self.CURRENT_PREGNANCY_CHOICES).keys():
            raise forms.ValidationError(
                "Current pregnancy must be a valid choice (0-6)."
            )
        return current_pregnancy

    def clean_marital_status(self):
        marital_status = int(self.cleaned_data.get('marital_status'))
        if marital_status not in dict(self.MARITAL_STATUS_CHOICES).keys():
            raise forms.ValidationError(
                "Select a valid marital status choice (0-2)."
            )
        return marital_status

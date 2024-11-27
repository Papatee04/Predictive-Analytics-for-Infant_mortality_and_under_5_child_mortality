from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator

class MortalityPredictionForm(forms.Form):
    """
    Form for collecting mortality prediction inputs
    """
    EDUCATION_CHOICES = [
        (0, 'No Education'),
        (1, 'Primary'),
        (2, 'Secondary'),
        (3, 'Higher')
    ]
    
    RESIDENCE_CHOICES = [
        (1, 'Urban'),
        (2, 'Rural')
    ]
    
    WEALTH_CHOICES = [
        (1, 'Poorest'),
        (2, 'Poor'),
        (3, 'Middle'),
        (4, 'Rich'),
        (5, 'Richest')
    ]
    
    REGION_CHOICES = [
        (1, 'Central'),
        (2, 'North'),
        (3, 'South'),
        (4, 'East'),
        (5, 'West'),
        # Add other regions as needed
    ]
    
    WATER_SOURCE_CHOICES = [
        (1, 'Improved Water Source'),
        (2, 'Unimproved Water Source')
    ]
    
    education_level = forms.ChoiceField(
        choices=EDUCATION_CHOICES, 
        help_text="Highest educational level"
    )
    
    residence_type = forms.ChoiceField(
        choices=RESIDENCE_CHOICES, 
        help_text="Type of residence"
    )
    
    wealth_index = forms.ChoiceField(
        choices=WEALTH_CHOICES, 
        help_text="Household wealth index"
    )
    
    region = forms.ChoiceField(
        choices=REGION_CHOICES, 
        help_text="Region of residence"
    )
    
    water_source = forms.ChoiceField(
        choices=WATER_SOURCE_CHOICES, 
        help_text="Primary source of drinking water"
    )
    
    postnatal_check = forms.BooleanField(
        required=False, 
        help_text="Was a postnatal check performed?"
    )
    succeeding_birth_interval = forms.IntegerField(
        required=False,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Succeeding birth interval in months (set to 0 if not applicable)."
    )
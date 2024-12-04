from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator

class MortalityPredictionForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom classes to form fields
        for field_name, field in self.fields.items():
            if isinstance(field, forms.ChoiceField):
                field.widget.attrs.update({
                    'class': 'block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500'
                })
            elif isinstance(field, forms.BooleanField):
                field.widget.attrs.update({
                    'class': 'h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded'
                })
            elif isinstance(field, forms.IntegerField):
                field.widget.attrs.update({
                    'class': 'block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500'
                })
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
        (2, 'Copperbelt'),
        (3, 'Eastern'),
        (4, 'Luapula'),
        (5, 'Lusaka'),
        (6, 'Muchinga'),
        (7, 'Northern'),
        (8, 'North Western'),
        (7, 'Southern'),
        (8, 'Western'),
    ]
    
    WATER_SOURCE_CHOICES = [
    # Improved Water Sources
    (10, 'Piped Water'),
    (11, 'Piped into dwelling'),
    (12, 'Piped to yard/plot'),
    (13, 'Piped to neighbor'),
    (14, 'Public tap/standpipe'),
    (20, 'Tube well water'),
    (21, 'Tube well or borehole'),
    (31, 'Protected well'),
    (41, 'Protected spring'),
    (51, 'Rainwater'),
    (71, 'Bottled water'),  # Bottled water can be considered improved if used exclusively and not from unimproved sources.

    # Unimproved Water Sources
    (32, 'Unprotected well'),
    (42, 'Unprotected spring'),
    (43, 'Surface water (e.g., river, dam, lake, pond, stream, canal, irrigation channel)'),
    (61, 'Tanker truck'),
    (62, 'Cart with small tank'),
    (96, 'Other'),
    (97, 'Not a de jure resident'),

    # Missing or unspecified
    (99, 'Missing'),
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
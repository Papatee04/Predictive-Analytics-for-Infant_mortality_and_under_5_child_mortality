from django.db import models


class ChildMortalityForm(models.Model):
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

    cause_of_fistula = models.IntegerField(
        choices=CAUSE_OF_FISTULA_CHOICES,
        help_text="Cause of fistula"
    )
    number_of_children = models.IntegerField(
        help_text="Number of living children"
    )
    current_pregnancy = models.IntegerField(
        choices=CURRENT_PREGNANCY_CHOICES,
        help_text="Living children + Current Pregnancy"
    )
    marital_status = models.IntegerField(
        choices=MARITAL_STATUS_CHOICES,
        help_text="Marital status"
    )
    use_family_planning = models.BooleanField(
        default=False,
        help_text="Using family planning"
    )
    education_years = models.IntegerField(
        help_text="Years of education"
    )
    age_at_first_sex = models.IntegerField(
        choices=[(i, f'{i} years') for i in range(
            7, 50)] + [(0, 'Not had sex'), (96, 'At first union'), (98, 'Don\'t know')],
        help_text="Age at first sexual encounter"
    )

    def __str__(self):
        return f"Form Response by {self.id}"

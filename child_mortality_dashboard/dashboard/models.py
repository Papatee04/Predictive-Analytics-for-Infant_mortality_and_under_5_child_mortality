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

    AGE_AT_FIRST_SEX_CHOICES = [(i, f'{i} years') for i in range(7, 50)] + [
        (0, 'Not had sex'),
        (96, 'At first union'),
        (98, 'Don\'t know'),
    ]

    USE_FAMILY_PLANNING_CHOICES = [
        (0, 'No'),
        (1, 'Yes'),
    ]

    cause_of_fistula = models.IntegerField(
        choices=CAUSE_OF_FISTULA_CHOICES,
        help_text="Cause of fistula"
    )
    number_of_children = models.IntegerField(
        help_text="Number of living children"
    )
    number_of_living_children = models.IntegerField(
        help_text="Number of living children + current pregnancy"
    )

    current_pregnancy = models.IntegerField(
        choices=CURRENT_PREGNANCY_CHOICES,
        help_text="Living children + Current Pregnancy"
    )
    marital_status = models.IntegerField(
        choices=MARITAL_STATUS_CHOICES,
        help_text="Marital status"
    )
    age_at_first_sex = models.IntegerField(
        choices=AGE_AT_FIRST_SEX_CHOICES,
        help_text="Age at first sexual encounter"
    )
    entries_in_birth_history = models.IntegerField(
        help_text="Entries in birth history"
    )
    total_children_ever_born = models.IntegerField(
        help_text="Total children ever born"
    )
    ever_been_married = models.IntegerField(
        choices=[(0, 'No'), (1, 'Formerly married'), (2, 'Lived with a man')],
        help_text="Ever been married or in union"
    )

    def __str__(self):
        return f"Form Response by {self.id}"

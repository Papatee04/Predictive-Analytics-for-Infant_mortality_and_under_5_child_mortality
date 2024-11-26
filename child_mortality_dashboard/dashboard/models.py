from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator


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
    
class ChildMortalityAssessment(models.Model):
    # Link to the form that was used for prediction
    form_data = models.ForeignKey('ChildMortalityForm', on_delete=models.CASCADE)
    
    # User who made the assessment
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Binary risk prediction
    risk_prediction = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="0 for low risk, 1 for high risk"
    )
    
    # Timestamp of assessment
    assessment_date = models.DateTimeField(auto_now_add=True)
    
    # Optional fields for tracking
    status = models.CharField(max_length=20, choices=[
        ('active', 'Active Monitoring'),
        ('resolved', 'Resolved'),
        ('closed', 'Closed')
    ], default='active')
    
    intervention_outcome = models.CharField(max_length=20, choices=[
        ('successful', 'Successful'),
        ('ongoing', 'Ongoing'),
        ('unsuccessful', 'Unsuccessful')
    ], null=True, blank=True)
    
    def __str__(self):
        return f"Assessment {self.id} - Risk: {'High' if self.risk_prediction == 1 else 'Low'}"
    
    def save(self, *args, **kwargs):
        # Automatically set status based on risk prediction
        if self.risk_prediction == 1:
            self.status = 'active'
        else:
            self.status = 'resolved'
        
        super().save(*args, **kwargs)

    class Meta:
        verbose_name_plural = "Child Mortality Assessments"    

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

    
class ChildMortalityAssessment(models.Model): 
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

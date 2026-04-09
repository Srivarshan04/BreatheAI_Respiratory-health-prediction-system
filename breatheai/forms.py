from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired

class HealthLogForm(FlaskForm):
    symptoms = TextAreaField('Symptoms', validators=[DataRequired()])
    medications = StringField('Medications')
    submit = SubmitField('Save Log')

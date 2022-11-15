from django import forms
from .models import sentimen
# creating a form 
class InputForm(forms.Form):
    cari = forms.CharField(label='Search', max_length=100)

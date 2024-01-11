from django import forms
from .models import Userinfo

class LoginForm(forms.ModelForm):
    subsr = forms.IntegerField()
    class Meta:
        model = Userinfo
        fields = ['subsr']
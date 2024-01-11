from .models import Userinfo
from rest_framework import serializers
# from .models import *
#import random

class SingupSerializer(serializers.ModelSerializer):
    class Meta:
        model = Userinfo
        fields = '__all__'
    def create(self, validated_data):
        subsr = validated_data.get('subsr')
        ip = validated_data.get('ip')
       
       #random.seed(id)
       #check = randint(1, 3000000000)

    #    x_fowarded_for = validated_data.META.get('HTTP_X_FORWARDED_FOR')
    #    if x_fowarded_for:
    #       ip = x_fowarded_for.split(',')[0]
    #    else:
    #        ip = validated_data.META.get('REMOTE_ADDR')

        user = Userinfo(
            subsr = subsr, 
            ip = ip
            #check = check
            )
        user.save()
        return user

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Userinfo
        fields = '__all__'
    def create(self, validated_data):
        return Userinfo(**validated_data)
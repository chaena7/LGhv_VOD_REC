#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import boto3

def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "base.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

REGION_NAME = 'ap-northeast-2'
BUCKET_NAME = 'hello00.net-model'
MODEL_FILE_PATH = 'service/ml_models/'
#


def download_model(obj):

    # if not os.path.isdir('service/ml_models'):
    #
        # os.mkdir('service/ml_models')
    os.makedirs('service/ml_models',  exist_ok=True)

    if not os.path.isfile(f'service/ml_models/{obj}'):
        s3 = boto3.client(
            's3',
            aws_access_key_id = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            region_name = REGION_NAME
        )
    # bucket = s3.Bucket(BUCKET_NAME)
        s3.download_file(BUCKET_NAME, obj, MODEL_FILE_PATH + obj)
        print(f'successfully download model{obj}')


if __name__ == "__main__":
    #get/ download model from s3
    download_model('model_genre.pkl')
    download_model('model_lightfm.pkl')

    from service.model_code import GenreBasedRecommendationModel, LightFM_Model
    main()

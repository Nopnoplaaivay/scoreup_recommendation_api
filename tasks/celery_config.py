import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=os.getenv("CELERY_RESULT_BACKEND"),
        broker=os.getenv("CELERY_BROKER_URL")
    )
    celery.conf.update(app.config)
    return celery
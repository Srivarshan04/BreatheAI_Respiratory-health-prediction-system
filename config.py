# config.py
import os

SQLALCHEMY_DATABASE_URI = "postgresql://username:password@localhost:5432/yourdbname"
MODEL_DIR = os.path.join("instance", "models")
SECRET_KEY = "your_secret_key"

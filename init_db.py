from app import app
from extensions import db
from models import User

with app.app_context():
    db.create_all()
    print("Database and User table created!")

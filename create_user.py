from app import app, db, User
from werkzeug.security import generate_password_hash

# Replace with your actual credentials
email = "kapilkushwahahathras2004@gmail.com"
password = "yourpassword123"

with app.app_context():
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        print("User already exists.")
    else:
        user = User(email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        print("User created successfully.")
# import secrets
# print(secrets.token_hex(16))

from werkzeug.security import generate_password_hash
from app.models import Admin
from app import create_app, db

app = create_app()

with app.app_context():
    admin = Admin(
        username='admin',
        password=generate_password_hash('Admin123!')
    )
    db.session.add(admin)
    db.session.commit()
    print("Admin user created successfully!")



# models.py
from main import db



# Tabla intermedia para la relación muchos a muchos entre usuarios y roles
user_roles = db.Table('user_roles',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('role_id', db.Integer, db.ForeignKey('role.id'), primary_key=True)
)

# Modelo de Usuario
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    roles = db.relationship('Role', secondary=user_roles, backref=db.backref('users', lazy='dynamic'))
    details = db.relationship('UserDetails3', backref='user', uselist=False)  # Relación 1:1 con UserDetails


class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)


# Modelo de Detalles de Usuario
class UserDetails3(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=False, nullable=False)
    gender = db.Column(db.String(50))
    car_owner = db.Column(db.String(50))
    property_owner = db.Column(db.String(50))
    children = db.Column(db.String(50))
    annual_income = db.Column(db.String(50))
    type_income = db.Column(db.String(50))
    education = db.Column(db.String(50))
    marital_status = db.Column(db.String(50))
    housing_type = db.Column(db.String(50))
    birthday_count = db.Column(db.String(50))
    employed_days = db.Column(db.String(50))
    mobile_phone = db.Column(db.String(50))
    work_phone = db.Column(db.String(50))
    phone = db.Column(db.String(50))
    email_id = db.Column(db.String(50))
    type_occupation = db.Column(db.String(50))
    family_members = db.Column(db.String(50))
    label = db.Column(db.String(50))
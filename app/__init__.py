from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bootstrap import Bootstrap
from flask_login import LoginManager

login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = 'auth.login'


db = SQLAlchemy()
migrate = Migrate()
bootstrap = Bootstrap()

def create_app(config_class=None):
    if config_class is None:
        from config import DevelopmentConfig
        config_class = DevelopmentConfig

    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    bootstrap.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    with app.app_context():
        from app import models
        db.create_all()

    # Register blueprints
    from .routes import main
    app.register_blueprint(main)

    from .auth import auth
    app.register_blueprint(auth, url_prefix='/auth')

    return app

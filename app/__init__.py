from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class=None):
    if config_class is None:
        from config import DevelopmentConfig
        config_class = DevelopmentConfig

    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)

    migrate.init_app(app, db)

    with app.app_context():
        from app import models
        db.create_all()

    # Register blueprints
    from .routes import main
    app.register_blueprint(main)

    return app

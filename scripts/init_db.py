"""
Initialize missing database tables from the existing project models and the
Milestone 2/3 models.

Run from the repository root:
    python -m scripts.init_db

This uses SQLAlchemy create_all: it creates missing tables but does NOT alter
existing tables. Inspect the current database schema and back it up first.
"""
from app.database import Base, engine
from app import models  # existing users, patients, imaging_cases
from app import models_milestone2_3  # register the new model classes

def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database table initialization completed.")
    print("Existing tables were not altered.")

if __name__ == "__main__":
    main()

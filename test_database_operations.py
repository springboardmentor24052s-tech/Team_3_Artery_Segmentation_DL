
from datetime import date
from sqlalchemy import select

from app.database import SessionLocal
from app.models import User, Patient, ImagingCase


def test_database_operations():
    db = SessionLocal()

    try:
        # Start a transaction
        test_email = "db_test_temp@example.com"

        user = User(
            email=test_email,
            password_hash="temporary_test_hash",
            full_name="Temporary DB Test",
            role="medical_researcher",
        )
        db.add(user)
        db.flush()

        patient = Patient(
            patient_code="TEMP-DB-TEST-001",
            age=45,
            gender="Unknown",
            created_by=user.id,
        )
        db.add(patient)
        db.flush()

        case = ImagingCase(
            patient_id=patient.id,
            created_by=user.id,
            scan_date=date.today(),
            modality="CT",
        )
        db.add(case)
        db.flush()

        # Verify records can be read
        saved_user = db.scalar(
            select(User).where(User.email == test_email)
        )
        saved_patient = db.scalar(
            select(Patient).where(
                Patient.patient_code == "TEMP-DB-TEST-001"
            )
        )
        saved_case = db.get(ImagingCase, case.id)

        assert saved_user is not None
        assert saved_patient is not None
        assert saved_case is not None

        print("USER INSERT/READ: PASS")
        print("PATIENT INSERT/READ: PASS")
        print("CASE INSERT/READ: PASS")
        print("FOREIGN KEY RELATIONSHIPS: PASS")

    except Exception:
        db.rollback()
        raise

    finally:
        # Always undo the test transaction
        db.rollback()
        db.close()


if __name__ == "__main__":
    test_database_operations()

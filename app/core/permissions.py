
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.routers.auth import get_current_user
from app.models import Image, ImagingCase, Patient, User


def check_patient_access(patient: Patient, user: User) -> None:
    if user.role != "administrator" and patient.created_by != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access this patient record",
        )


def check_case_access(case: ImagingCase, user: User) -> None:
    if user.role != "administrator" and case.created_by != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access this imaging case",
        )


def get_accessible_case(
    db: Session,
    case_id: str,
    user: User,
) -> ImagingCase:
    case = db.get(ImagingCase, case_id)

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Imaging case not found",
        )

    check_case_access(case, user)
    return case


def get_accessible_image(
    db: Session,
    image_id: str,
    user: User,
) -> Image:
    image = db.get(Image, image_id)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found",
        )

    get_accessible_case(db, image.case_id, user)
    return image


def require_roles(*allowed_roles: str):
    def role_checker(
        user: User = Depends(get_current_user),
    ) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource",
            )

        return user

    return role_checker

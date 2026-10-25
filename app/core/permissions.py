
from fastapi import Depends, HTTPException, status

from app.routers.auth import get_current_user
from app.models import User


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

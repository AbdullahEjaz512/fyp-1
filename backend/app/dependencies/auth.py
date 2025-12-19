from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database import get_db, User
from app.services.auth_service import AuthService
from app.services.firebase_auth import firebase_auth_client


# Local security/authorization utilities isolated to avoid circular imports
security = HTTPBearer()
auth_service = AuthService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """
    Verify Firebase ID token or fallback JWT and return a minimal user dict.
    Extracted to dependencies/auth.py to prevent circular imports with app.main
    and routers importing each other.
    """
    token = credentials.credentials

    # Try Firebase first
    try:
        decoded_firebase = firebase_auth_client.verify_id_token(token)
        user = firebase_auth_client.sync_firebase_user(decoded_firebase, db)
        return {
            "user_id": str(user.user_id),
            "email": user.email,
            "username": user.username,
            "role": user.role,
        }
    except Exception:
        pass  # Not a valid Firebase token, attempt JWT fallback

    # Fallback: project JWT
    try:
        payload = auth_service.decode_access_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )

        db_user = db.query(User).filter(User.user_id == int(user_id)).first()
        if db_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        return {
            "user_id": str(db_user.user_id),
            "email": db_user.email,
            "username": db_user.username,
            "role": db_user.role,
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


async def get_current_doctor(user = Depends(get_current_user)):
    """Ensure current user has a clinical role (doctor/radiologist/oncologist/admin)."""
    if user.get("role") not in ["doctor", "radiologist", "oncologist", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access this resource",
        )
    return user

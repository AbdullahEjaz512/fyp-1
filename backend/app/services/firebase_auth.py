"""
Firebase Authentication helper for Module 10
- Verifies Firebase ID tokens
- Syncs Firebase users into local database with role claims
- Allows setting custom claims (doctor/admin) using service credentials
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import firebase_admin
from firebase_admin import auth, credentials
from sqlalchemy.orm import Session

from app.services.auth_service import AuthService
from app.database import User

logger = logging.getLogger(__name__)


class FirebaseAuthClient:
    def __init__(self, credentials_path: Optional[str] = None, storage_bucket: Optional[str] = None):
        self.credentials_path = credentials_path or os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
        self.storage_bucket = storage_bucket or os.getenv("FIREBASE_STORAGE_BUCKET")
        self.default_role = os.getenv("FIREBASE_DEFAULT_ROLE", "doctor")
        self._ensure_app()

    def _ensure_app(self):
        if firebase_admin._apps:
            return firebase_admin.get_app()

        cred_path = Path(self.credentials_path)
        if cred_path.exists():
            cred = credentials.Certificate(str(cred_path))
            logger.info("✓ Firebase initialized with service account at %s", cred_path)
        else:
            cred = credentials.ApplicationDefault()
            logger.warning("⚠ Firebase credentials not found at %s, using Application Default Credentials", cred_path)

        options = {"storageBucket": self.storage_bucket} if self.storage_bucket else None
        return firebase_admin.initialize_app(cred, options)

    def verify_id_token(self, id_token: str) -> Dict:
        self._ensure_app()
        decoded = auth.verify_id_token(id_token)
        return decoded

    def set_role_claim(self, uid: str, role: str) -> Dict[str, str]:
        self._ensure_app()
        role_value = role.lower()
        if role_value not in ["doctor", "admin", "radiologist", "oncologist", "researcher", "patient"]:
            role_value = self.default_role
        auth.set_custom_user_claims(uid, {"role": role_value})
        logger.info("✓ Firebase custom claim set for uid=%s role=%s", uid, role_value)
        return {"uid": uid, "role": role_value}

    def set_role_by_email(self, email: str, role: str) -> Dict[str, str]:
        self._ensure_app()
        user_record = auth.get_user_by_email(email)
        return self.set_role_claim(user_record.uid, role)

    def sync_firebase_user(self, decoded_token: Dict, db: Session) -> User:
        """Ensure Firebase user exists in local Postgres with matching role"""
        email = decoded_token.get("email")
        uid = decoded_token.get("uid")
        name = decoded_token.get("name") or (email.split("@", 1)[0] if email else uid)

        # Prefer custom claim 'role', then token claim 'role'
        role = decoded_token.get("role") or decoded_token.get("claims", {}).get("role")
        if not role:
            role = decoded_token.get("firebase", {}).get("sign_in_provider")
        if role not in ["admin", "doctor", "radiologist", "oncologist", "researcher", "patient"]:
            role = self.default_role

        if not email:
            raise ValueError("Firebase token missing email; cannot sync user")

        user = db.query(User).filter(User.email == email).first()
        if not user:
            username_seed = email.split("@", 1)[0]
            username = username_seed
            suffix = 1
            while db.query(User).filter(User.username == username).first():
                username = f"{username_seed}{suffix}"
                suffix += 1

            placeholder_password = AuthService.hash_password(AuthService.generate_session_id())
            user = User(
                email=email,
                username=username,
                password_hash=placeholder_password,
                role=role,
                full_name=name,
                medical_license=None,
                specialization=None,
                institution=None,
                department=None,
                years_of_experience=0,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info("✓ Created local user from Firebase uid=%s email=%s", uid, email)
        else:
            if role and user.role != role:
                user.role = role
                db.commit()
                db.refresh(user)
                logger.info("✓ Updated local user role for email=%s to %s", email, role)

        return user


firebase_auth_client = FirebaseAuthClient()

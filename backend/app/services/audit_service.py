"""
Audit logging service for Module 10 (Cloud Infrastructure and Security)
- Records access to sensitive resources (patient data, scans, AI results)
- Stores entries in the audit_logs table and writes to audit logger
"""

from datetime import datetime
from typing import Any, Dict, Optional
import logging

from app.database import SessionLocal, AuditLog

SENSITIVE_PATH_PREFIXES = (
    "/api/v1/files",
    "/api/v1/analysis",
    "/api/v1/auth/me",
    "/api/v1/reconstruction",
    "/api/v1/assistant",
    "/api/v1/visualization",
    "/api/v1/security",
)


class AuditLogger:
    """Simple audit logger that writes to DB and Python logger"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger("audit")

    def should_log(self, path: str) -> bool:
        return self.enabled and any(path.startswith(prefix) for prefix in SENSITIVE_PATH_PREFIXES)

    def log_request(
        self,
        *,
        user: Optional[Dict[str, Any]],
        path: str,
        method: str,
        status_code: int,
        ip_address: Optional[str],
        user_agent: Optional[str],
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.should_log(path):
            return

        session = SessionLocal()
        try:
            entry = AuditLog(
                user_id=str(user.get("user_id")) if user else None,
                role=user.get("role") if user else None,
                action=f"{method} {path}",
                resource_type=resource_type,
                resource_id=resource_id,
                method=method,
                path=path,
                status_code=status_code,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=datetime.utcnow(),
                extra=extra or {},
            )
            session.add(entry)
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            self.logger.error("Audit logging failed: %s", exc)
        finally:
            session.close()

        # Also push to application log for quick visibility
        self.logger.info(
            "AUDIT user=%s role=%s path=%s status=%s ip=%s",
            user.get("user_id") if user else "anonymous",
            user.get("role") if user else "unknown",
            path,
            status_code,
            ip_address,
        )

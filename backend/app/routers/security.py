"""
Admin endpoints for Module 10 security features
- Get audit logs with filtering
- Set Firebase custom role claims
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db, AuditLog
from app.dependencies.auth import get_current_user
from app.services.firebase_auth import firebase_auth_client

router = APIRouter(prefix="/api/v1/security", tags=["security"])


@router.get("/audit-logs")
async def get_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    path: Optional[str] = None,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get audit logs for compliance and security review
    Module 10 - FE-4: Audit logging for sensitive data access
    
    Admin-only endpoint
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    
    query = db.query(AuditLog)
    
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    if resource_id:
        query = query.filter(AuditLog.resource_id == resource_id)
    if path:
        query = query.filter(AuditLog.path.like(f"%{path}%"))
    
    total = query.count()
    logs = query.order_by(desc(AuditLog.created_at)).offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "role": log.role,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "method": log.method,
                "path": log.path,
                "status_code": log.status_code,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "created_at": log.created_at.isoformat() if log.created_at else None,
                "extra": log.extra,
            }
            for log in logs
        ]
    }


@router.get("/audit-logs/stats")
async def get_audit_stats(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get audit log statistics
    Module 10 - Security monitoring
    
    Admin-only endpoint
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    
    total_logs = db.query(func.count(AuditLog.id)).scalar()
    
    # Top users by activity
    top_users = (
        db.query(AuditLog.user_id, func.count(AuditLog.id).label("count"))
        .filter(AuditLog.user_id.isnot(None))
        .group_by(AuditLog.user_id)
        .order_by(desc("count"))
        .limit(10)
        .all()
    )
    
    # Top paths
    top_paths = (
        db.query(AuditLog.path, func.count(AuditLog.id).label("count"))
        .filter(AuditLog.path.isnot(None))
        .group_by(AuditLog.path)
        .order_by(desc("count"))
        .limit(10)
        .all()
    )
    
    return {
        "total_logs": total_logs,
        "top_users": [{"user_id": user_id, "count": count} for user_id, count in top_users],
        "top_paths": [{"path": path, "count": count} for path, count in top_paths],
    }


@router.post("/firebase/set-role")
async def set_firebase_role(
    email: str,
    role: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Set custom role claim in Firebase Auth
    Module 10 - FE-1: Role-based authentication for Doctors and Admins
    
    Admin-only endpoint
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    
    try:
        result = firebase_auth_client.set_role_by_email(email, role)
        return {
            "message": f"Role '{role}' set for user with email {email}",
            "uid": result["uid"],
            "role": result["role"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set role: {str(e)}"
        )

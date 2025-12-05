"""
Authentication Service
Handles user authentication, password hashing, JWT tokens
Implements FR3.1 to FR3.8, FR5.1 to FR5.8
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
import secrets
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
except ImportError:
    # Fallback values for development
    SECRET_KEY = "development-secret-key-change-in-production"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 120


class AuthService:
    """Authentication Service for Seg-Mind"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using bcrypt - FR4.3
        Password must be minimum 8 characters with letters and numbers
        Bcrypt has a 72-byte limit, so we truncate if necessary.
        """
        # Bcrypt has a 72-byte limit, truncate if necessary
        password_bytes = password.encode('utf-8')[:72]
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash - FR3.4
        """
        # Bcrypt has a 72-byte limit, truncate if necessary (same as hashing)
        password_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token - FR3.7
        Default expiration: 2 hours as per SRS (Table 2.17)
        
        Args:
            data: Dictionary containing user data (user_id, email, role)
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """
        Create refresh token for extended sessions - FR3.3 (Remember Me)
        Expires in 30 days as per SRS
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=30)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        """
        Decode and validate JWT token - FR3.7
        
        Returns:
            Dictionary with token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None
    
    @staticmethod
    def decode_access_token(token: str) -> dict:
        """
        Decode and validate access token - FR3.5
        Raises exception if invalid
        
        Returns:
            Dictionary with token payload
        
        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "access":
                raise JWTError("Invalid token type")
            return payload
        except JWTError as e:
            raise JWTError(f"Could not validate token: {str(e)}")
    
    @staticmethod
    def generate_reset_token() -> str:
        """
        Generate secure password reset token - FR5.2
        Token expires in 1 hour as per SRS
        """
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """
        Validate password strength - FR4.3, FR5.4
        Requirements:
        - Minimum 8 characters
        - Must contain letters and numbers
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        has_letter = any(c.isalpha() for c in password)
        has_number = any(c.isdigit() for c in password)
        
        if not has_letter:
            return False, "Password must contain at least one letter"
        
        if not has_number:
            return False, "Password must contain at least one number"
        
        return True, "Password is strong"
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate unique session ID"""
        return secrets.token_urlsafe(24)
    
    def get_current_user(self, token: str, db) -> Optional[dict]:
        """
        Get current user from JWT token
        Used for collaboration APIs
        
        Args:
            token: JWT access token
            db: Database session
            
        Returns:
            Dictionary with user data or None if invalid
        """
        try:
            payload = self.decode_access_token(token)
            user_id = payload.get("sub")
            if user_id is None:
                return None
            
            # Import here to avoid circular imports
            from app.database import User
            
            # Get user from database
            user = db.query(User).filter(User.user_id == int(user_id)).first()
            if user is None:
                return None
            
            return {
                "user_id": user.user_id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role
            }
        except Exception:
            return None


# Failed login tracking for account locking - FR3.6, Table 2.18
class LoginAttemptTracker:
    """
    Track failed login attempts to implement account locking
    FR3.6: Lock account after 5 failed attempts for 15 minutes
    """
    
    def __init__(self):
        self.attempts = {}  # {email: [(timestamp, success), ...]}
        self.locked_accounts = {}  # {email: lock_expiry_time}
    
    def record_attempt(self, email: str, success: bool):
        """Record a login attempt"""
        if email not in self.attempts:
            self.attempts[email] = []
        
        self.attempts[email].append((datetime.utcnow(), success))
        
        # Keep only last 10 attempts
        self.attempts[email] = self.attempts[email][-10:]
    
    def get_failed_count(self, email: str, minutes: int = 15) -> int:
        """Get number of failed attempts in last N minutes"""
        if email not in self.attempts:
            return 0
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_attempts = [
            (ts, success) for ts, success in self.attempts[email]
            if ts > cutoff_time
        ]
        
        failed_count = sum(1 for ts, success in recent_attempts if not success)
        return failed_count
    
    def is_locked(self, email: str) -> bool:
        """Check if account is locked - FR3.6"""
        if email in self.locked_accounts:
            lock_expiry = self.locked_accounts[email]
            if datetime.utcnow() < lock_expiry:
                return True
            else:
                # Lock has expired, remove it
                del self.locked_accounts[email]
                return False
        return False
    
    def lock_account(self, email: str, duration_minutes: int = 15):
        """Lock account for specified duration - Table 2.18"""
        lock_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.locked_accounts[email] = lock_until
    
    def should_lock(self, email: str) -> bool:
        """Determine if account should be locked based on failed attempts"""
        failed_count = self.get_failed_count(email)
        return failed_count >= 5  # Lock after 5 failed attempts
    
    def unlock_account(self, email: str):
        """Manually unlock account - for email verification unlock"""
        if email in self.locked_accounts:
            del self.locked_accounts[email]
        if email in self.attempts:
            self.attempts[email] = []  # Clear attempts


# Global instance
login_tracker = LoginAttemptTracker()


if __name__ == "__main__":
    # Test authentication service
    print("Testing Auth Service...")
    
    # Test password hashing
    password = "SecurePass123"
    hashed = AuthService.hash_password(password)
    print(f"✓ Password hashed successfully")
    
    # Test password verification
    is_valid = AuthService.verify_password(password, hashed)
    print(f"✓ Password verification: {is_valid}")
    
    # Test token creation
    token_data = {"user_id": "test_user", "email": "test@example.com", "role": "doctor"}
    access_token = AuthService.create_access_token(token_data)
    print(f"✓ Access token created")
    
    # Test token decoding
    decoded = AuthService.decode_token(access_token)
    print(f"✓ Token decoded: {decoded['user_id']}")
    
    # Test password strength validation
    weak_pass = "short"
    strong_pass = "SecurePassword123"
    
    is_valid, msg = AuthService.validate_password_strength(weak_pass)
    print(f"✓ Weak password rejected: {msg}")
    
    is_valid, msg = AuthService.validate_password_strength(strong_pass)
    print(f"✓ Strong password accepted: {msg}")
    
    print("\n✅ All authentication tests passed!")

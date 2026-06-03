"""
Session Management Service

Handles session state persistence across the 6-page pipeline.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

import json
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions for the multi-page processing pipeline"""

    def __init__(self, storage_dir: str = "uploads/sessions"):
        """
        Initialize session manager

        Args:
            storage_dir: Directory for session storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SessionManager initialized with storage: {self.storage_dir}")

    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())

    def create_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Create a new session

        Args:
            session_id: Unique session identifier
            data: Initial session data
        """
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata
        data["session_id"] = session_id
        data["created_at"] = datetime.now().isoformat()
        data["updated_at"] = datetime.now().isoformat()

        session_file = session_dir / "session.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Created session: {session_id}")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary

        Raises:
            ValueError: If session not found
        """
        session_file = self.storage_dir / session_id / "session.json"
        if not session_file.exists():
            raise ValueError(f"Session {session_id} not found")

        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update session data

        Args:
            session_id: Session identifier
            updates: Data to merge into session
        """
        session_data = self.get_session(session_id)
        session_data.update(updates)
        session_data["updated_at"] = datetime.now().isoformat()

        session_file = self.storage_dir / session_id / "session.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Updated session: {session_id}")

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session and all its files

        Args:
            session_id: Session identifier
        """
        session_dir = self.storage_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up sessions older than max_age_hours

        Args:
            max_age_hours: Maximum session age in hours

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0

        for session_dir in self.storage_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                created_at = datetime.fromisoformat(data["created_at"])
                if created_at < cutoff:
                    shutil.rmtree(session_dir)
                    deleted_count += 1
                    logger.info(f"Cleaned up old session: {session_dir.name}")
            except Exception as e:
                logger.error(f"Error cleaning session {session_dir.name}: {e}")

        logger.info(f"Cleanup complete: {deleted_count} sessions deleted")
        return deleted_count

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return (self.storage_dir / session_id / "session.json").exists()

    def get_session_path(self, session_id: str) -> Path:
        """Get the filesystem path for a session"""
        return self.storage_dir / session_id

from datetime import datetime, timedelta
import json

import pytest

from src.web.services.session_manager import SessionManager


@pytest.fixture
def manager(tmp_path):
    return SessionManager(storage_dir=str(tmp_path / "sessions"))


def test_create_then_get_returns_data_with_metadata(manager):
    manager.create_session("abc", {"current_step": 1})

    session = manager.get_session("abc")

    assert session["current_step"] == 1
    assert session["session_id"] == "abc"
    assert "created_at" in session and "updated_at" in session


def test_get_missing_session_raises_value_error(manager):
    with pytest.raises(ValueError):
        manager.get_session("missing")


def test_update_merges_and_keeps_existing_keys(manager):
    manager.create_session("abc", {"a": 1})

    manager.update_session("abc", {"b": 2})

    session = manager.get_session("abc")
    assert session["a"] == 1 and session["b"] == 2


def test_delete_removes_session(manager):
    manager.create_session("abc", {})

    manager.delete_session("abc")

    assert not manager.session_exists("abc")


def test_cleanup_removes_only_expired_sessions(manager):
    manager.create_session("old", {})
    manager.create_session("new", {})
    old_file = manager.get_session_path("old") / "session.json"
    data = json.loads(old_file.read_text())
    data["created_at"] = (datetime.now() - timedelta(hours=48)).isoformat()
    old_file.write_text(json.dumps(data))

    deleted = manager.cleanup_old_sessions(max_age_hours=24)

    assert deleted == 1
    assert not manager.session_exists("old")
    assert manager.session_exists("new")


def test_generated_session_ids_are_unique(manager):
    assert manager.generate_session_id() != manager.generate_session_id()

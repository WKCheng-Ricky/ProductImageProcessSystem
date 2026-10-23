from datetime import datetime, timedelta
import json

import pytest

from src.web.services.session_manager import SessionManager


SESSION_ID = "11111111-1111-4111-8111-111111111111"
OLD_ID = "33333333-3333-4333-8333-333333333333"
NEW_ID = "batch_0a1b2c3d"


@pytest.fixture
def manager(tmp_path):
    return SessionManager(storage_dir=str(tmp_path / "sessions"))


def test_create_then_get_returns_data_with_metadata(manager):
    manager.create_session(SESSION_ID, {"current_step": 1})

    session = manager.get_session(SESSION_ID)

    assert session["current_step"] == 1
    assert session["session_id"] == SESSION_ID
    assert "created_at" in session and "updated_at" in session


def test_get_missing_session_raises_value_error(manager):
    with pytest.raises(ValueError):
        manager.get_session("22222222-2222-4222-8222-222222222222")


def test_update_merges_and_keeps_existing_keys(manager):
    manager.create_session(SESSION_ID, {"a": 1})

    manager.update_session(SESSION_ID, {"b": 2})

    session = manager.get_session(SESSION_ID)
    assert session["a"] == 1 and session["b"] == 2


def test_delete_removes_session(manager):
    manager.create_session(SESSION_ID, {})

    manager.delete_session(SESSION_ID)

    assert not manager.session_exists(SESSION_ID)


def test_cleanup_removes_only_expired_sessions(manager):
    manager.create_session(OLD_ID, {})
    manager.create_session(NEW_ID, {})
    old_file = manager.get_session_path(OLD_ID) / "session.json"
    data = json.loads(old_file.read_text())
    data["created_at"] = (datetime.now() - timedelta(hours=48)).isoformat()
    old_file.write_text(json.dumps(data))

    deleted = manager.cleanup_old_sessions(max_age_hours=24)

    assert deleted == 1
    assert not manager.session_exists(OLD_ID)
    assert manager.session_exists(NEW_ID)


def test_generated_session_ids_are_unique(manager):
    assert manager.generate_session_id() != manager.generate_session_id()


@pytest.mark.parametrize("bad_id", ["../x", "..", "a/b", "abc", "", "batch_../x", "11111111-1111-4111-8111-11111111111/"])
def test_invalid_ids_are_rejected_everywhere(manager, bad_id):
    with pytest.raises(ValueError):
        manager.create_session(bad_id, {})
    with pytest.raises(ValueError):
        manager.get_session(bad_id)
    with pytest.raises(ValueError):
        manager.delete_session(bad_id)
    with pytest.raises(ValueError):
        manager.get_session_path(bad_id)
    assert not manager.session_exists(bad_id)


def test_delete_with_traversal_id_keeps_other_directories(manager, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()

    with pytest.raises(ValueError):
        manager.delete_session("../victim")

    assert victim.exists()


def test_batch_prefixed_id_is_accepted(manager):
    manager.create_session("batch_0a1b2c3d", {})

    assert manager.session_exists("batch_0a1b2c3d")

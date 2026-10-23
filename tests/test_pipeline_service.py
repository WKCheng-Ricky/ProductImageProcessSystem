import threading
import time

import pytest

from src.web.services import pipeline_service
from src.web.services.pipeline_service import PipelineService


@pytest.fixture(autouse=True)
def reset_service():
    PipelineService.reset_pipeline()
    yield
    PipelineService.reset_pipeline()


def test_concurrent_first_access_builds_the_pipeline_once(monkeypatch):
    builds = []

    def slow_build():
        builds.append(1)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(PipelineService, "_build_pipeline", staticmethod(slow_build))

    results = []
    threads = [threading.Thread(target=lambda: results.append(pipeline_service.get_pipeline())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(builds) == 1
    assert len({id(r) for r in results}) == 1


def test_failed_build_is_not_cached(monkeypatch):
    attempts = []

    def flaky_build():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return object()

    monkeypatch.setattr(PipelineService, "_build_pipeline", staticmethod(flaky_build))

    with pytest.raises(RuntimeError):
        pipeline_service.get_pipeline()

    assert pipeline_service.get_pipeline() is not None

"""BrowseService: which stored images the Browse & Search page offers."""
import random

import numpy as np
import pytest

from src.pipeline.product_grouper import InMemoryVectorDB, ProductGroupUUID
from src.web.services.browse_service import BrowseService

DIMENSION = 16


def one_hot(index, dimension=DIMENSION):
    vector = np.zeros(dimension)
    vector[index] = 1.0
    return vector


def add(db, image_id, vector, with_url=True, **metadata):
    if with_url:
        metadata["image_url"] = f"/img/{image_id}.jpg"
    db.insert(image_id, vector, metadata)


def distinct_db(count, dimension=DIMENSION):
    db = InMemoryVectorDB()
    for index in range(count):
        add(db, f"img{index}", one_hot(index, dimension))
    return db


def service(db, seed=0, groups=()):
    return BrowseService(db, list_groups=lambda: list(groups), rng=random.Random(seed))


def test_random_mode_returns_count_distinct_images():
    sample = service(distinct_db(15)).sample(10)

    ids = [image.id for image in sample.images]
    assert sample.mode == "random"
    assert len(ids) == 10 and len(set(ids)) == 10
    assert sample.total_in_db == 15
    assert sample.ungrouped_count is None


def test_fewer_images_than_count_returns_all():
    assert len(service(distinct_db(3)).sample(10).images) == 3


def test_empty_db_returns_no_images():
    sample = service(InMemoryVectorDB()).sample(10)

    assert sample.images == []
    assert sample.total_in_db == 0


def test_near_identical_images_collapse_to_one_tile():
    db = InMemoryVectorDB()
    for index in range(30):
        add(db, f"copy{index}", one_hot(0))

    assert len(service(db).sample(10).images) == 1


def test_images_without_image_url_are_never_returned():
    db = InMemoryVectorDB()
    for index in range(5):
        add(db, f"with{index}", one_hot(index))
        add(db, f"without{index}", one_hot(index + 5), with_url=False)

    ids = {image.id for image in service(db).sample(10).images}

    assert ids and all(image_id.startswith("with") for image_id in ids)


def test_a_db_where_no_record_has_a_url_gives_no_images_but_a_total():
    db = InMemoryVectorDB()
    for index in range(5):
        add(db, f"bare{index}", one_hot(index), with_url=False)

    sample = service(db).sample(10)

    assert sample.images == []
    assert sample.total_in_db == 5


def test_zero_norm_embedding_does_not_crash():
    db = distinct_db(1)
    db.embeddings["zero"] = np.zeros(DIMENSION)
    db.metadata["zero"] = {"image_url": "/img/zero.jpg"}

    ids = {image.id for image in service(db).sample(10).images}

    assert "img0" in ids


def test_sampling_never_loads_all_embeddings():
    db = distinct_db(5)
    db.get_all_embeddings = lambda: pytest.fail("loaded every vector")

    service(db).sample(10)


def test_same_seed_gives_same_sample_and_different_seeds_can_differ():
    db = distinct_db(40, dimension=40)

    first = [image.id for image in service(db, seed=1).sample(5).images]
    again = [image.id for image in service(db, seed=1).sample(5).images]
    other = [image.id for image in service(db, seed=2).sample(5).images]

    assert first == again
    assert first != other


def group(group_id, member_ids, representative_id=None):
    return ProductGroupUUID(group_id=group_id, member_ids=list(member_ids), representative_id=representative_id)


def test_grouped_mode_returns_one_image_per_group():
    db = distinct_db(6)
    groups = [group("g1", ["img0", "img1"]), group("g2", ["img2", "img3"]), group("g3", ["img4", "img5"])]

    sample = service(db, groups=groups).sample(10)

    ids = {image.id for image in sample.images}
    assert sample.mode == "grouped"
    assert len(sample.images) == 3
    assert all(len(ids & set(g.member_ids)) == 1 for g in groups)


def test_best_member_is_the_highest_overall_score():
    db = InMemoryVectorDB()
    for index, overall in enumerate([0.2, 0.9, 0.5]):
        add(db, f"m{index}", one_hot(index), overall=overall)

    sample = service(db, groups=[group("g", ["m0", "m1", "m2"])]).sample(10)

    assert [image.id for image in sample.images] == ["m1"]


def test_quality_score_is_used_when_overall_is_absent():
    db = InMemoryVectorDB()
    add(db, "low", one_hot(0), quality_score=0.3)
    add(db, "high", one_hot(1), quality_score=0.8)

    sample = service(db, groups=[group("g", ["low", "high"])]).sample(10)

    assert [image.id for image in sample.images] == ["high"]


def test_unscored_group_uses_representative_id():
    db = distinct_db(2)

    sample = service(db, groups=[group("g", ["img0", "img1"], representative_id="img1")]).sample(10)

    assert [image.id for image in sample.images] == ["img1"]


def test_dead_representative_falls_back_to_a_live_member():
    db = distinct_db(1)

    sample = service(db, groups=[group("g", ["img0", "gone"], representative_id="gone")]).sample(10)

    assert [image.id for image in sample.images] == ["img0"]


def test_dead_members_and_emptied_groups_are_skipped():
    db = distinct_db(2)
    groups = [group("dead", ["gone1", "gone2"]), group("live", ["img0", "img1"], representative_id="img0")]

    sample = service(db, groups=groups).sample(10)

    assert [image.id for image in sample.images] == ["img0"]
    assert sample.mode == "grouped"


def test_best_member_without_url_falls_back_to_the_next_member():
    db = InMemoryVectorDB()
    add(db, "best", one_hot(0), with_url=False, overall=0.9)
    add(db, "next", one_hot(1), overall=0.4)
    add(db, "bare1", one_hot(2), with_url=False)
    add(db, "bare2", one_hot(3), with_url=False)
    groups = [group("g1", ["best", "next"]), group("g2", ["bare1", "bare2"])]

    sample = service(db, groups=groups).sample(10)

    assert [image.id for image in sample.images] == ["next"]


def test_fewer_groups_than_count_returns_every_group():
    db = distinct_db(4)
    groups = [group("g1", ["img0", "img1"]), group("g2", ["img2", "img3"])]

    assert len(service(db, groups=groups).sample(10).images) == 2


def test_more_groups_than_count_returns_exactly_count():
    db = distinct_db(8)
    groups = [group(f"g{i}", [f"img{i}"]) for i in range(8)]

    assert len(service(db, groups=groups).sample(5).images) == 5


def test_groups_that_miss_some_images_are_stale_so_sampling_is_random():
    db = distinct_db(5)

    sample = service(db, groups=[group("g", ["img0", "img1", "img2", "gone"])]).sample(10)

    assert sample.mode == "random"
    assert sample.total_in_db == 5
    assert sample.ungrouped_count == 2
    assert {image.id for image in sample.images} == {f"img{i}" for i in range(5)}


def test_groups_covering_every_image_report_nothing_ungrouped():
    db = distinct_db(4)

    sample = service(db, groups=[group("g1", ["img0", "img1"]), group("g2", ["img2", "img3"])]).sample(10)

    assert sample.mode == "grouped"
    assert sample.ungrouped_count == 0


def test_all_groups_dead_falls_back_to_random_mode():
    db = distinct_db(3)

    sample = service(db, groups=[group("g", ["gone1", "gone2"])]).sample(10)

    assert sample.mode == "random"
    assert sample.ungrouped_count is None
    assert len(sample.images) == 3


def test_failing_list_groups_falls_back_to_random_mode():
    def broken():
        raise RuntimeError("groups db is down")

    sample = BrowseService(distinct_db(3), list_groups=broken, rng=random.Random(0)).sample(10)

    assert sample.mode == "random"
    assert len(sample.images) == 3

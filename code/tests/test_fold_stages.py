"""Initial preparation highlights do not invent a logged cleaning action."""

from maus_export.fold_stages import initial_cleaning_end


def test_known_preparation_prefix_has_a_display_interval() -> None:
    assert (
        initial_cleaning_end(
            [
                (0, "init"),
                (2, "prepare"),
                (4, "check_serving_plate"),
                (8, "oil"),
                (12, "spread_batter"),
            ]
        )
        == 8
    )


def test_no_cleaning_is_inferred_when_video_starts_in_production() -> None:
    assert initial_cleaning_end([(0, "cook"), (10, "add_topping")]) == 0
    assert initial_cleaning_end([]) == 0
    assert initial_cleaning_end([(0, "init")]) == 0
    assert initial_cleaning_end([(0, "wipe_spatula"), (5, "cook")]) == 0

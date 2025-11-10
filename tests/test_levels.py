from nested_learning.levels import LevelClock, LevelSpec


def test_level_clock_updates_on_schedule() -> None:
    specs = [LevelSpec(name="fast", update_period=1), LevelSpec(name="slow", update_period=3)]
    clock = LevelClock(specs)
    updates = []
    for step in range(5):
        for spec in specs:
            if clock.should_update(spec.name):
                updates.append((step, spec.name))
                clock.record_update(spec.name)
        clock.tick()
    assert updates[0] == (0, "fast")
    assert any(level == "slow" for _, level in updates)

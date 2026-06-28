from pathlib import Path

from vlawm.tennis.data import label_from_name, train_test_split_clips, Clip


def test_label_from_name_forehand():
    assert label_from_name("forehand_03.MOV") == 0


def test_label_from_name_backhand():
    assert label_from_name("backhand_12.mov") == 1


def test_label_from_name_rejects_unknown():
    try:
        label_from_name("serve_01.MOV")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown label")


def _clip(name, label):
    return Clip(path=Path(name), label=label, name=name)


def test_split_holds_out_one_per_class_deterministically():
    clips = [_clip(f"forehand_{i}.MOV", 0) for i in range(5)]
    clips += [_clip(f"backhand_{i}.MOV", 1) for i in range(6)]
    train, test = train_test_split_clips(clips, seed=0)
    assert len(test) == 2
    assert {c.label for c in test} == {0, 1}
    assert len(train) == 9
    # determinism
    train2, test2 = train_test_split_clips(clips, seed=0)
    assert [c.name for c in test] == [c.name for c in test2]
    # no leakage
    test_names = {c.name for c in test}
    assert all(c.name not in test_names for c in train)

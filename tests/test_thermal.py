from clb.thermal import cool_down


def test_cool_box_does_not_wait():
    assert cool_down(read=lambda: 60, sleep=lambda s: None, log=lambda m: None) == 0.0


def test_hot_box_waits_until_resume_temperature():
    temps = iter([80, 76, 72, 69])
    slept = []
    w = cool_down(hot=78, resume=70, read=lambda: next(temps), sleep=slept.append, log=lambda m: None)
    assert w == 15 and len(slept) == 3

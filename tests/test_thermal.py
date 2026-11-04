from clb.thermal import cool_down


def test_cool_box_does_not_wait():
    assert cool_down(read=lambda: 60, sleep=lambda s: None, log=lambda m: None) == 0.0


def test_hot_box_waits_until_resume_temperature():
    temps = iter([80, 76, 72, 69])
    slept = []
    w = cool_down(hot=78, resume=70, read=lambda: next(temps), sleep=slept.append, log=lambda m: None)
    assert w == 15 and len(slept) == 3


def test_rests_as_long_as_the_previous_work_spell():
    from clb import thermal
    clock, slept = [100.0], []
    kw = dict(read=lambda: 50, sleep=slept.append, now=lambda: clock[0], log=lambda m: None)
    thermal._last[0] = None
    thermal.cool_down(**kw)          # first call: nothing to rest from
    clock[0] += 12
    thermal.cool_down(**kw)          # 12 s of work -> 12 s rest
    clock[0] += 500
    thermal.cool_down(**kw)          # long gap (not GPU work) -> no rest
    assert slept == [12]

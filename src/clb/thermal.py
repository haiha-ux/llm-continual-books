"""Thermostat for this GB10 box: pause work while the hottest ACPI zone is too hot.

This machine reaches 85 C on its ACPI zones after ~36 s of GPU load at only ~25 W and has hard-frozen (silent
power cut, no kernel log) four times under sustained LLM decode. Callers invoke `cool_down()` between batches /
training steps; it returns immediately when the box is cool and otherwise sleeps until it has cooled.
"""
import glob
import time

HOT_C = 76     # start pausing at this hottest-zone temperature
RESUME_C = 72  # resume once cooled to this
DUTY_CAP_S = 30  # work spells longer than this are rested for DUTY_CAP_S only
_last = [None]  # time the previous spell of work started (end of the previous cool_down call)


def hottest_c():
    t = 0
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            t = max(t, int(open(f).read()) // 1000)
        except (OSError, ValueError):
            pass
    return t


def cool_down(hot=HOT_C, resume=RESUME_C, log=print, read=hottest_c, sleep=time.sleep, now=time.monotonic):
    """Rest as long as the work since the previous call took (50 % duty cycle, at most DUTY_CAP_S), then block while
    the box is hot. Every GPU loop calls this once per unit of work. Returns seconds spent waiting for the cool-down."""
    if _last[0] is not None and now() - _last[0] < 120:
        sleep(min(now() - _last[0], DUTY_CAP_S))
    try:
        return _wait(hot, resume, log, read, sleep)
    finally:
        _last[0] = now()


def _wait(hot, resume, log, read, sleep):
    t = read()
    if t < hot:
        return 0.0
    waited = 0.0
    log(f"thermal: {t} C >= {hot} C, pausing until <= {resume} C")
    while t > resume:
        sleep(5)
        waited += 5
        t = read()
    log(f"thermal: resumed at {t} C after {waited:.0f} s")
    return waited

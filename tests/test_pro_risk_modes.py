from risk_manager import RiskManager


def _warmup_normal(rm: RiskManager, nav: float = 100_000.0, days: int = 40):
    for _ in range(days):
        rm.evaluate(nav=nav, daily_return=0.0005)


def test_instant_downgrade_to_emergency():
    rm = RiskManager()
    _warmup_normal(rm)

    # EMERGENCY threshold is DD <= -35%; use nav=60k => -40% DD
    snapshot = rm.evaluate(nav=60_000.0, daily_return=0.0)
    assert snapshot["mode"] == "EMERGENCY", f"Expected EMERGENCY, got {snapshot['mode']}"


def test_emergency_recovery_requires_5_consecutive_days():
    rm = RiskManager()
    _warmup_normal(rm)
    rm.evaluate(nav=60_000.0, daily_return=0.0)
    assert rm.current_mode.value == "EMERGENCY"

    # Keep NAV under rolling peak to preserve negative DD while vol/var normalize
    # EMERGENCY recovery thresholds: DD > -28%, vol < 120%, var < 6%
    for i in range(4):
        snapshot = rm.evaluate(nav=75_000.0, daily_return=0.0)
        assert snapshot["mode"] == "EMERGENCY", f"Unexpected early recovery on day {i+1}"
        assert snapshot["recovery_counter"] < 5

    snapshot = rm.evaluate(nav=75_000.0, daily_return=0.0)
    assert snapshot["mode"] == "RISK_OFF", f"Expected RISK_OFF after 5 days, got {snapshot['mode']}"
    assert snapshot["recovery_counter"] == 0


def test_recovery_counter_resets_on_single_breach():
    rm = RiskManager()
    _warmup_normal(rm)
    rm.evaluate(nav=60_000.0, daily_return=0.0)
    assert rm.current_mode.value == "EMERGENCY"

    # 3 valid recovery days
    for _ in range(3):
        snapshot = rm.evaluate(nav=75_000.0, daily_return=0.0)
    assert snapshot["recovery_counter"] == 3

    # One breach (drop back into EMERGENCY) should reset counter to 0
    snapshot = rm.evaluate(nav=60_000.0, daily_return=0.0)
    assert snapshot["recovery_counter"] == 0
    assert snapshot["mode"] == "EMERGENCY"


if __name__ == "__main__":
    test_instant_downgrade_to_emergency()
    test_emergency_recovery_requires_5_consecutive_days()
    test_recovery_counter_resets_on_single_breach()
    print("test_pro_risk_modes.py: PASS")

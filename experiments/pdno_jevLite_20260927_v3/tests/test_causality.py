from pdno.data.causal import ObservationEvent, latest_available


def test_late_arrival_is_not_backfilled_into_past_decision():
    events = [
        ObservationEvent(4, 4, "on_time"),
        ObservationEvent(5, 8, "late_newer_frame"),
        ObservationEvent(2, 3, "older"),
    ]
    assert latest_available(events, 6).payload_ref == "on_time"


def test_capture_and_receive_must_both_precede_decision():
    events = [ObservationEvent(7, 7, "future"), ObservationEvent(3, 7, "not_received")]
    assert latest_available(events, 6) is None


def test_empty_causal_history_returns_none():
    assert latest_available([], 0) is None

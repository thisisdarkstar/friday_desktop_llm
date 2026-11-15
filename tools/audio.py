"""Audio control tools using modern pycaw (Windows Core Audio API)."""

from pycaw.pycaw import AudioUtilities


def _get_volume_endpoint():
    speakers = AudioUtilities.GetSpeakers()
    if speakers is None:
        raise RuntimeError("No audio output device found.")
    return speakers.EndpointVolume


def set_volume(level: int) -> dict:
    """Set master audio volume level from 0 to 100."""
    level = max(0, min(100, int(level)))
    endpoint = _get_volume_endpoint()
    endpoint.SetMasterVolumeLevelScalar(level / 100.0, None)
    return {"status": "success", "level": level}


def get_volume() -> dict:
    """Get current master audio volume level and mute status."""
    endpoint = _get_volume_endpoint()
    scalar = endpoint.GetMasterVolumeLevelScalar()
    level = int(round(scalar * 100))
    muted = bool(endpoint.GetMute())
    return {"status": "success", "level": level, "muted": muted}


def mute_audio(mute: bool = True) -> dict:
    """Mute or unmute master audio."""
    endpoint = _get_volume_endpoint()
    endpoint.SetMute(1 if mute else 0, None)
    return {"status": "success", "muted": mute}


def toggle_mute() -> dict:
    """Toggle current mute state."""
    endpoint = _get_volume_endpoint()
    current = bool(endpoint.GetMute())
    endpoint.SetMute(0 if current else 1, None)
    return {"status": "success", "muted": not current}


def change_volume(delta: int) -> dict:
    """Increase or decrease volume by delta percentage points."""
    curr = get_volume()["level"]
    new_level = max(0, min(100, curr + delta))
    return set_volume(new_level)

class NullDMXController:
    """
    Fallback DMX controller used when the real USB DMX device or driver is not
    available. The runtime can keep running because these methods are no-ops.
    """

    def update_channel(self, *args, **kwargs):
        return None

    def run(self, *args, **kwargs):
        return None

    def close(self):
        return None


_dmx_warning_logged = False


def create_dmx_controller(port, device_type="ftdi"):
    """
    Try to create the real DMX controller and fall back to a no-op
    implementation if the dependency or USB device is unavailable.
    """
    global _dmx_warning_logged

    try:
        from pyDMXController import pyDMXController

        return pyDMXController(port=port, device_type=device_type)
    except Exception as exc:
        if not _dmx_warning_logged:
            print(f"DMX controller unavailable. Running without DMX output: {exc}")
            _dmx_warning_logged = True
        return NullDMXController()

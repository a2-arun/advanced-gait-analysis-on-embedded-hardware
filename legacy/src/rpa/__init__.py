"""Intentionally empty for now.

The event interface (src/events/identity_event.py) is the actual RPA
integration point - it already writes structured JSON events and can POST
to a webhook. This package is reserved for a specific RPA platform
integration (UiPath/Power Automate/etc) once one is chosen; per the
project's own phasing, that choice comes after the event interface is
proven out, not before.
"""

"""Check every Home Assistant API the Murdock integration relies on.

Import-level and signature-level verification against an installed HA, so
a wrong assumption surfaces here instead of on someone's live system.
The repo's own test suite can't cover ``custom_components/`` — Home
Assistant needs Python 3.14 while the add-on image is on 3.11 — hence
this standalone check.

Run it in a throwaway container::

    docker build -t murdock-ha-verify - <<'EOF'
    FROM python:3.14-slim
    RUN apt-get update -qq && apt-get install -y -qq build-essential
    RUN pip install --no-cache-dir homeassistant
    EOF

    docker run --rm -v "$(pwd):/repo:ro" -w /repo murdock-ha-verify \
        python scripts/verify_integration_api.py

Exits non-zero on the first broken assumption. Extend it whenever the
integration starts using a new Home Assistant API.
"""

import importlib
import inspect
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.simplefilter("always", DeprecationWarning)

from homeassistant.const import __version__ as ha_version

print(f"=== HA {ha_version} ===\n")

ok = fail = 0


def check(label, fn):
    global ok, fail
    try:
        result = fn()
    except Exception as exc:
        fail += 1
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
    else:
        ok += 1
        print(f"ok    {label}" + (f"  → {result}" if result else ""))


def attr_of(module_path, name):
    def _inner():
        mod = importlib.import_module(module_path)
        if not hasattr(mod, name):
            raise AttributeError(f"{module_path} has no {name}")
        return None
    return _inner


# --- module-level imports of the integration itself -------------------
for mod in (
    "const", "api", "coordinator", "llm_api", "vocabulary",
    "entity", "sensor", "binary_sensor", "config_flow", "helpers",
    "__init__",
):
    check(
        f"import custom_components.murdock.{mod}",
        lambda m=mod: importlib.import_module(f"custom_components.murdock.{m}") and None,
    )

print()

# --- individual APIs --------------------------------------------------
check("exposed_entities.async_should_expose", attr_of(
    "homeassistant.components.homeassistant.exposed_entities",
    "async_should_expose"))
check("entity_registry.EVENT_ENTITY_REGISTRY_UPDATED", attr_of(
    "homeassistant.helpers.entity_registry", "EVENT_ENTITY_REGISTRY_UPDATED"))
check("area_registry.EVENT_AREA_REGISTRY_UPDATED", attr_of(
    "homeassistant.helpers.area_registry", "EVENT_AREA_REGISTRY_UPDATED"))
check("floor_registry.EVENT_FLOOR_REGISTRY_UPDATED", attr_of(
    "homeassistant.helpers.floor_registry", "EVENT_FLOOR_REGISTRY_UPDATED"))
check("area_registry.AreaRegistry.async_list_areas", attr_of(
    "homeassistant.helpers.area_registry", "AreaRegistry"))
check("floor_registry.FloorRegistry.async_list_floors", attr_of(
    "homeassistant.helpers.floor_registry", "FloorRegistry"))
check("llm.async_register_api", attr_of("homeassistant.helpers.llm",
                                        "async_register_api"))
check("llm.APIInstance", attr_of("homeassistant.helpers.llm", "APIInstance"))
check("dt_util.utc_from_timestamp", attr_of("homeassistant.util.dt",
                                            "utc_from_timestamp"))
check("event.async_track_time_interval", attr_of(
    "homeassistant.helpers.event", "async_track_time_interval"))
check("event.async_call_later", attr_of("homeassistant.helpers.event",
                                        "async_call_later"))
check("device_registry.DeviceInfo", attr_of(
    "homeassistant.helpers.device_registry", "DeviceInfo"))

print()

# --- the ones most likely to have moved -------------------------------
def entity_category_location():
    from homeassistant.const import EntityCategory as FromConst
    try:
        from homeassistant.helpers.entity import EntityCategory as FromHelpers
    except ImportError:
        return "const only — helpers.entity import BROKEN"
    return "both work" if FromConst is FromHelpers else "DIFFERENT objects!"


check("EntityCategory import path", entity_category_location)


def registry_methods():
    from homeassistant.helpers import area_registry as ar, floor_registry as fr
    missing = []
    for cls, meths in (
        (ar.AreaRegistry, ["async_list_areas", "async_get_area"]),
        (fr.FloorRegistry, ["async_list_floors", "async_get_floor"]),
    ):
        for m in meths:
            if not hasattr(cls, m):
                missing.append(f"{cls.__name__}.{m}")
    if missing:
        raise AttributeError(", ".join(missing))
    return None


check("registry list/get methods", registry_methods)


def llm_signatures():
    from homeassistant.helpers import llm
    ctx_fields = set(getattr(llm.LLMContext, "__dataclass_fields__", {}))
    if "device_id" not in ctx_fields:
        raise AttributeError("LLMContext lacks device_id")
    api_init = inspect.signature(llm.API.__init__).parameters
    for p in ("hass", "id", "name"):
        if p not in api_init:
            raise AttributeError(f"llm.API.__init__ lacks {p}")
    inst = inspect.signature(llm.APIInstance.__init__).parameters
    for p in ("api", "api_prompt", "llm_context", "tools"):
        if p not in inst:
            raise AttributeError(f"APIInstance lacks {p}")
    return f"LLMContext fields: {sorted(ctx_fields)}"


check("llm.API / APIInstance / LLMContext", llm_signatures)


def selector_options_as_strings():
    from homeassistant.helpers.selector import (
        SelectSelector, SelectSelectorConfig, SelectSelectorMode,
    )
    SelectSelector(
        SelectSelectorConfig(
            options=["clean", "inline"],
            mode=SelectSelectorMode.DROPDOWN,
            translation_key="context_mode",
        )
    )
    return None


check("SelectSelector with plain-string options", selector_options_as_strings)


def options_flow_shape():
    from homeassistant.config_entries import OptionsFlow
    # Setting config_entry explicitly is deprecated; we rely on the
    # property being provided by the framework.
    if not isinstance(getattr(OptionsFlow, "config_entry", None), property):
        raise AttributeError("OptionsFlow.config_entry is not a property")
    if not hasattr(OptionsFlow, "async_show_menu"):
        raise AttributeError("no async_show_menu")
    return None


check("OptionsFlow.config_entry property + async_show_menu", options_flow_shape)


def sensor_classes():
    from homeassistant.components.sensor import SensorDeviceClass
    from homeassistant.components.binary_sensor import BinarySensorDeviceClass
    SensorDeviceClass.TIMESTAMP
    BinarySensorDeviceClass.CONNECTIVITY
    return None


check("Sensor/BinarySensor device classes", sensor_classes)


def build_snapshot_signature():
    from custom_components.murdock.vocabulary import build_snapshot
    return str(inspect.signature(build_snapshot))


check("vocabulary.build_snapshot importable", build_snapshot_signature)


def mqtt_subscribe_api():
    """The MQTT path is how token-free setups deliver recognitions."""
    from homeassistant.components import mqtt
    sig = inspect.signature(mqtt.async_subscribe).parameters
    for p in ("hass", "topic", "msg_callback"):
        if p not in sig:
            raise AttributeError(f"async_subscribe lacks {p}")
    if not inspect.iscoroutinefunction(mqtt.async_subscribe):
        raise TypeError("async_subscribe is not awaitable")
    from homeassistant.components.mqtt.models import ReceiveMessage
    for f in ("topic", "payload"):
        if f not in ReceiveMessage.__dataclass_fields__:
            raise AttributeError(f"ReceiveMessage lacks {f}")
    return None


check("mqtt.async_subscribe + ReceiveMessage", mqtt_subscribe_api)


def payload_applies_from_both_paths():
    """One recognition, two transports, identical resulting state."""
    from custom_components.murdock.coordinator import MurdockCoordinator

    payload = {
        "satellite_id": "assist_satellite.arbeitszimmer",
        "speaker": "Jonas",
        "is_known": True,
        "confidence": 0.94,
        "distance": 0.2142,
        "threshold": 0.38,
        "nearest_speaker": "Alex",
        "nearest_distance": 0.61,
        "weight": 1.0,
        "timestamp": 1785000000.5,
        "ambiguities": [
            {"original": "Bad", "alternative": "Bett", "kind": "alternative"}
        ],
    }

    class FakeCoordinator(MurdockCoordinator):
        def __init__(self):  # bypass HA plumbing
            self._states = {}
            self._unsubs = []
            self._last_seen = None
            self.mqtt_subscribed = False
            self.available = False
            self.last_event_at = None
            self.hass = None

        def _dispatch(self, sat):
            pass

    import custom_components.murdock.coordinator as mod

    sent = []
    original = mod.async_dispatcher_send
    mod.async_dispatcher_send = lambda hass, sig, sat: sent.append(sat)
    try:
        # Event-bus path
        c = FakeCoordinator()
        c._apply_payload(payload, source="event")
        via_event = c._states[payload["satellite_id"]]
        # MQTT path: same payload, JSON-encoded
        import json as _json

        encoded = _json.dumps(payload)

        class Msg:
            topic = "murdock/event/recognition"
            payload = encoded

        c2 = FakeCoordinator()
        c2._handle_mqtt_message(Msg())
        via_mqtt = c2._states[payload["satellite_id"]]
        # Duplicate suppression on a single coordinator
        c3 = FakeCoordinator()
        c3._apply_payload(payload, source="event")
        c3._handle_mqtt_message(Msg())
        dispatches = len(sent)
    finally:
        mod.async_dispatcher_send = original

    if via_event.speaker != "Jonas" or via_mqtt.speaker != "Jonas":
        raise AssertionError("speaker not applied on both paths")
    if via_mqtt.weight != 1.0 or via_mqtt.nearest_distance != 0.61:
        raise AssertionError("fields lost on the MQTT path")
    if len(via_mqtt.ambiguities) != 1:
        raise AssertionError("ambiguities lost on the MQTT path")
    # 1 (event) + 1 (mqtt) + 1 (third coordinator, duplicate suppressed)
    if dispatches != 3:
        raise AssertionError(f"expected 3 dispatches, got {dispatches}")
    return "event + mqtt identical, duplicate suppressed"


check("recognition applies from event bus and MQTT", payload_applies_from_both_paths)

print(f"\n=== {ok} ok, {fail} failed ===")
sys.exit(1 if fail else 0)

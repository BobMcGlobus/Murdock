"""The Murdock integration against a real Home Assistant core.

Not part of the add-on's test suite: Home Assistant needs Python 3.14,
the add-on image is 3.11. In a Linux container::

    docker run --rm -v "$(pwd):/repo" -w /repo python:3.14-slim sh -c       "pip install -q pytest-homeassistant-custom-component hassil          home-assistant-intents ha-ffmpeg mutagen paho-mqtt &&        python -m pytest -q -o asyncio_mode=auto tests_ha"

On Windows, in a venv with the same packages, put the stand-ins for the
Linux-only modules on the path first::

    PYTHONPATH=tests_ha/winstubs python -m pytest -q -o asyncio_mode=auto tests_ha
"""

from __future__ import annotations

import aiohttp
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

DOMAIN = "murdock"
OLD_URL = "http://homeassistant.local:8099"
INTERNAL = "http://9a87bc67-murdock:8099"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    yield


@pytest.fixture(autouse=True)
async def core_components(hass: HomeAssistant):
    """What a real install has running before any integration loads."""
    assert await async_setup_component(hass, "homeassistant", {})


def _mock_proxy(aioclient_mock, base, *, version=None):
    aioclient_mock.get(f"{base}/api/version", json=version or {"version": "0.13.0"})
    aioclient_mock.get(f"{base}/api/state", json={"satellites": []})
    aioclient_mock.get(f"{base}/api/vocabulary", json={"available": False})
    aioclient_mock.get(f"{base}/api/satellites", json={"satellites": []})


def _entry(minor_version=2, url=OLD_URL):
    return MockConfigEntry(
        domain=DOMAIN,
        version=1,
        minor_version=minor_version,
        unique_id=url,
        data={
            "base_url": url,
            "token": None,
            "satellites": [
                {"satellite_id": "wz", "satellite_entity": "assist_satellite.wz"}
            ],
        },
        options={"mirror_vocabulary": False, "mqtt_prefix": ""},
    )


async def _setup(hass, entry):
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return hass.data[DOMAIN][entry.entry_id]


async def test_ids_are_keyed_by_entry_and_satellites_hang_under_the_proxy(
    hass: HomeAssistant, aioclient_mock, caplog
):
    _mock_proxy(aioclient_mock, OLD_URL)
    entry = _entry()
    await _setup(hass, entry)

    ents = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    assert ents, "no entities were created"
    assert all(e.unique_id.startswith(entry.entry_id + ":") for e in ents), \
        [e.unique_id for e in ents]

    devs = {
        next(iter(d.identifiers))[1]: d
        for d in dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    }
    proxy = devs[entry.entry_id]
    satellite = devs[f"{entry.entry_id}:wz"]
    assert satellite.via_device_id == proxy.id
    assert "via_device" not in caplog.text, "the deprecated parameter is still used"


async def test_old_url_keyed_ids_are_migrated_in_place(
    hass: HomeAssistant, aioclient_mock
):
    """Entity ids — and every automation using them — must survive."""
    _mock_proxy(aioclient_mock, OLD_URL)
    entry = _entry(minor_version=1)
    entry.add_to_hass(hass)
    ent_reg = er.async_get(hass)
    old = ent_reg.async_get_or_create(
        "binary_sensor", DOMAIN, f"{OLD_URL}:connection",
        config_entry=entry, suggested_object_id="murdock_proxy_verbindung",
    )
    dev_reg = dr.async_get(hass)
    dev_reg.async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={(DOMAIN, OLD_URL)},
        name="Murdock Proxy",
    )

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    migrated = ent_reg.async_get(old.entity_id)
    assert migrated is not None, "the entity was replaced, not migrated"
    assert migrated.unique_id == f"{entry.entry_id}:connection"
    assert entry.minor_version == 2
    assert dev_reg.async_get_device_by_identifier((DOMAIN, entry.entry_id), entry.entry_id)
    assert dev_reg.async_get_device_by_identifier((DOMAIN, OLD_URL), entry.entry_id) is None
    # No duplicate created next to it.
    conns = [e for e in er.async_entries_for_config_entry(ent_reg, entry.entry_id)
             if e.unique_id.endswith(":connection")]
    assert len(conns) == 1


async def test_one_failed_poll_does_not_mean_offline(hass: HomeAssistant, aioclient_mock):
    _mock_proxy(aioclient_mock, OLD_URL)
    entry = _entry()
    coordinator = await _setup(hass, entry)
    assert coordinator.available is True

    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{OLD_URL}/api/version", exc=aiohttp.ClientConnectionError("reset"))
    await coordinator._async_health_poll()
    await coordinator._async_health_poll()
    assert coordinator.available is True, "went offline after two blips"
    await coordinator._async_health_poll()
    assert coordinator.available is False

    aioclient_mock.clear_requests()
    _mock_proxy(aioclient_mock, OLD_URL)
    await coordinator._async_health_poll()
    assert coordinator.available is True


async def test_a_recognition_counts_as_a_sign_of_life(hass: HomeAssistant, aioclient_mock):
    _mock_proxy(aioclient_mock, OLD_URL)
    entry = _entry()
    coordinator = await _setup(hass, entry)
    coordinator._apply_payload(
        {"satellite_id": "wz", "speaker": "Jonas", "is_known": True, "timestamp": 1},
        source="mqtt",
    )
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{OLD_URL}/api/version", exc=aiohttp.ClientConnectionError("x"))
    for _ in range(5):
        await coordinator._async_health_poll()
    assert coordinator.available is True


async def test_the_entry_moves_to_the_internal_address(hass: HomeAssistant, aioclient_mock):
    _mock_proxy(aioclient_mock, OLD_URL, version={
        "version": "0.13.0", "internal_url": INTERNAL, "addon_slug": "9a87bc67_murdock",
    })
    _mock_proxy(aioclient_mock, INTERNAL)
    entry = _entry()
    coordinator = await _setup(hass, entry)
    await coordinator._async_health_poll()
    await hass.async_block_till_done()
    assert entry.data["base_url"] == INTERNAL
    # Same entities after the reload, not a second set.
    ents = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    assert len([e for e in ents if e.unique_id.endswith(":connection")]) == 1


async def test_an_unreachable_internal_address_is_left_alone(hass: HomeAssistant, aioclient_mock):
    _mock_proxy(aioclient_mock, OLD_URL, version={"version": "0.13.0", "internal_url": INTERNAL})
    aioclient_mock.get(f"{INTERNAL}/api/version", exc=aiohttp.ClientConnectionError("no route"))
    entry = _entry()
    coordinator = await _setup(hass, entry)
    await coordinator._async_health_poll()
    await hass.async_block_till_done()
    assert entry.data["base_url"] == OLD_URL

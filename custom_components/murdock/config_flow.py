"""Config flow for the Murdock integration (plan §16)."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)

from .api import MurdockApiClient, MurdockApiError
from .const import (
    CONF_BASE_URL,
    CONF_CONTEXT_MODE,
    CONF_FRESHNESS_WINDOW,
    CONF_MIRROR_VOCABULARY,
    CONF_MQTT_PREFIX,
    CONF_SATELLITE_ENTITY,
    CONF_SATELLITE_ID,
    CONF_SATELLITES,
    CONF_TOKEN,
    CONTEXT_MODES,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_FRESHNESS_WINDOW,
    DEFAULT_MIRROR_VOCABULARY,
    DEFAULT_MQTT_PREFIX,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

_CONF_ADD_ANOTHER = "add_another"


def _settings_schema(current: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(
                CONF_CONTEXT_MODE,
                default=current.get(CONF_CONTEXT_MODE, DEFAULT_CONTEXT_MODE),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=CONTEXT_MODES,
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key="context_mode",
                )
            ),
            vol.Required(
                CONF_FRESHNESS_WINDOW,
                default=current.get(
                    CONF_FRESHNESS_WINDOW, DEFAULT_FRESHNESS_WINDOW
                ),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=5, max=600, step=1, mode=NumberSelectorMode.BOX,
                    unit_of_measurement="s",
                )
            ),
            vol.Required(
                CONF_MIRROR_VOCABULARY,
                default=current.get(
                    CONF_MIRROR_VOCABULARY, DEFAULT_MIRROR_VOCABULARY
                ),
            ): BooleanSelector(),
            # Murdock's MQTT topic prefix. Recognitions arrive over MQTT
            # when you run the token-free setup; blank disables the MQTT
            # path and relies on the REST event bus alone.
            vol.Optional(
                CONF_MQTT_PREFIX,
                default=current.get(CONF_MQTT_PREFIX, DEFAULT_MQTT_PREFIX),
            ): TextSelector(),
        }
    )


def _satellite_schema(discovered: list[str]) -> vol.Schema:
    options = [
        SelectOptionDict(value=sat, label=sat) for sat in discovered
    ]
    return vol.Schema(
        {
            vol.Required(CONF_SATELLITE_ID): SelectSelector(
                SelectSelectorConfig(
                    options=options,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Required(CONF_SATELLITE_ENTITY): EntitySelector(
                EntitySelectorConfig(domain="assist_satellite")
            ),
            vol.Optional(_CONF_ADD_ANOTHER, default=False): BooleanSelector(),
        }
    )


class MurdockConfigFlow(ConfigFlow, domain=DOMAIN):
    """Connection → satellite mapping → settings."""

    VERSION = 1

    def __init__(self) -> None:
        self._base_url: str = ""
        self._token: str | None = None
        self._discovered: list[str] = []
        self._satellites: list[dict[str, str]] = []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            base_url = user_input[CONF_BASE_URL].strip().rstrip("/")
            token = (user_input.get(CONF_TOKEN) or "").strip() or None
            client = MurdockApiClient(self.hass, base_url, token)
            try:
                version = await client.get_version()
                sats = await client.get_satellites()
            except MurdockApiError:
                errors["base"] = "cannot_connect"
            else:
                await self.async_set_unique_id(base_url)
                self._abort_if_unique_id_configured()
                self._base_url = base_url
                self._token = token
                self._discovered = [s["satellite_id"] for s in sats]
                _LOGGER.debug(
                    "Connected to Murdock %s (%d satellites seen)",
                    version, len(self._discovered),
                )
                return await self.async_step_satellite()
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BASE_URL, default="http://homeassistant.local:8099"
                    ): TextSelector(),
                    vol.Optional(CONF_TOKEN): TextSelector(),
                }
            ),
            errors=errors,
        )

    async def async_step_satellite(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Map one satellite; repeat while "add another" is checked."""
        if user_input is not None:
            self._satellites.append(
                {
                    CONF_SATELLITE_ID: user_input[CONF_SATELLITE_ID],
                    CONF_SATELLITE_ENTITY: user_input[CONF_SATELLITE_ENTITY],
                }
            )
            if user_input.get(_CONF_ADD_ANOTHER):
                return await self.async_step_satellite()
            return await self.async_step_settings()
        remaining = [
            s for s in self._discovered
            if s not in {m[CONF_SATELLITE_ID] for m in self._satellites}
        ]
        return self.async_show_form(
            step_id="satellite",
            data_schema=_satellite_schema(remaining),
        )

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(
                title="Murdock",
                data={
                    CONF_BASE_URL: self._base_url,
                    CONF_TOKEN: self._token,
                    CONF_SATELLITES: self._satellites,
                },
                options=user_input,
            )
        return self.async_show_form(
            step_id="settings", data_schema=_settings_schema({})
        )

    @staticmethod
    @callback
    def async_get_options_flow(entry: ConfigEntry) -> MurdockOptionsFlow:
        return MurdockOptionsFlow()


class MurdockOptionsFlow(OptionsFlow):
    """Edit settings or remap satellites without re-adding the entry."""

    def __init__(self) -> None:
        self._satellites: list[dict[str, str]] = []

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return self.async_show_menu(
            step_id="init", menu_options=["settings", "satellites"]
        )

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(
                data={**self.config_entry.options, **user_input}
            )
        current = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(
            step_id="settings", data_schema=_settings_schema(current)
        )

    async def async_step_satellites(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Rebuild the satellite mapping from scratch."""
        if user_input is not None:
            self._satellites.append(
                {
                    CONF_SATELLITE_ID: user_input[CONF_SATELLITE_ID],
                    CONF_SATELLITE_ENTITY: user_input[CONF_SATELLITE_ENTITY],
                }
            )
            if user_input.get(_CONF_ADD_ANOTHER):
                return await self.async_step_satellites()
            data = {
                **self.config_entry.data,
                CONF_SATELLITES: self._satellites,
            }
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=data
            )
            return self.async_create_entry(data=dict(self.config_entry.options))
        client = MurdockApiClient(
            self.hass,
            self.config_entry.data[CONF_BASE_URL],
            self.config_entry.data.get(CONF_TOKEN),
        )
        discovered: list[str] = []
        try:
            discovered = [
                s["satellite_id"] for s in await client.get_satellites()
            ]
        except MurdockApiError:
            pass
        remaining = [
            s for s in discovered
            if s not in {m[CONF_SATELLITE_ID] for m in self._satellites}
        ]
        return self.async_show_form(
            step_id="satellites",
            data_schema=_satellite_schema(remaining),
        )

# Home Assistant Config Flow & Options Flow — Complete UI Reference

Research document covering all interface options available for building config flows and options flows in Home Assistant integrations.

## Table of Contents

- [Flow Result Types (Core Methods)](#flow-result-types-core-methods)
- [Forms (`async_show_form`)](#forms-async_show_form)
- [All 40 Selectors (UI Input Types)](#all-40-selectors-ui-input-types)
- [Multi-Step Wizards](#multi-step-wizards)
- [Menu Navigation (`async_show_menu`)](#menu-navigation-async_show_menu)
- [Progress / Long-Running Tasks (`async_show_progress`)](#progress--long-running-tasks-async_show_progress)
- [External Step Flows (OAuth2)](#external-step-flows-oauth2)
- [Collapsible Sections](#collapsible-sections)
- [Suggested Values vs Defaults](#suggested-values-vs-defaults)
- [Read-Only Fields](#read-only-fields)
- [Error Handling](#error-handling)
- [Description Placeholders & Translations](#description-placeholders--translations)
- [Specialized Flow Types](#specialized-flow-types)
- [Config Subentry Flows (HA 2025.3+)](#config-subentry-flows-ha-20253)
- [Schema-Based Flow Helper (Declarative)](#schema-based-flow-helper-declarative)
- [Preview Support](#preview-support)
- [Browser Autofill](#browser-autofill)
- [Icons](#icons)
- [Discovery Flows](#discovery-flows)
- [Entry Lifecycle Helpers](#entry-lifecycle-helpers)
- [Summary of All UI Patterns](#summary-of-all-ui-patterns)
- [Relevance to Area Occupancy Detection](#relevance-to-area-occupancy-detection)
- [Sources](#sources)

---

## Flow Result Types (Core Methods)

Every flow handler has these methods, each producing a different UI outcome:

| Method | UI Result |
|---|---|
| `async_show_form()` | Input form dialog |
| `async_show_menu()` | Clickable navigation menu |
| `async_show_progress()` | Spinner/progress bar |
| `async_external_step()` | Redirect to external site (OAuth) |
| `async_create_entry()` | Success, flow ends |
| `async_abort()` | Error/info message, flow ends |

### Complete Method Signatures

```python
# Show a form to gather user input
async def async_show_form(
    self,
    *,
    step_id: str | None = None,
    data_schema: vol.Schema | None = None,
    errors: dict[str, str] | None = None,
    description_placeholders: Mapping[str, str] | None = None,
    last_step: bool | None = None,
    preview: str | None = None,
) -> FlowResultT

# Create a config entry (finish the flow)
async def async_create_entry(
    self,
    *,
    title: str | None = None,
    data: Mapping[str, Any],
    description: str | None = None,
    description_placeholders: Mapping[str, str] | None = None,
) -> FlowResultT

# Abort the flow with a reason
async def async_abort(
    self,
    *,
    reason: str,
    description_placeholders: Mapping[str, str] | None = None,
) -> FlowResultT

# Redirect user to an external URL (OAuth2 etc.)
async def async_external_step(
    self,
    *,
    step_id: str | None = None,
    url: str,
    description_placeholders: Mapping[str, str] | None = None,
) -> FlowResultT

# Mark external step as complete
async def async_external_step_done(
    self,
    *,
    next_step_id: str,
) -> FlowResultT

# Show a progress spinner for long-running tasks
async def async_show_progress(
    self,
    *,
    step_id: str | None = None,
    progress_action: str,
    description_placeholders: Mapping[str, str] | None = None,
    progress_task: asyncio.Task[Any] | None = None,
) -> FlowResultT

# Update progress percentage (0.0 to 1.0)
async def async_update_progress(
    self,
    progress: float,
) -> None

# Mark progress as done, advance to next step
async def async_show_progress_done(
    self,
    *,
    next_step_id: str,
) -> FlowResultT

# Show a menu of navigational options
async def async_show_menu(
    self,
    *,
    step_id: str | None = None,
    menu_options: Container[str],
    sort: bool = False,
    description_placeholders: Mapping[str, str] | None = None,
) -> FlowResultT

# Called when a flow is removed/cancelled
async def async_remove(self) -> None
```

---

## Forms (`async_show_form`)

The primary UI method. Shows a dialog with input fields defined by `data_schema`.

### Parameters

- **`step_id`**: Identifies the current step. Determines which translation keys to use.
- **`data_schema`**: A `vol.Schema` defining form fields. Can use raw Voluptuous types or Selectors.
- **`errors`**: Dict mapping field names to translation error keys. Use `"base"` for form-level errors not tied to a specific field.
- **`description_placeholders`**: Dict of `{placeholder: value}` for substitution into translated title/description strings.
- **`last_step`**: Controls the submit button text. `True` = "Submit", `False` = "Next", `None` = auto-detect (HA guesses based on whether more steps follow).
- **`preview`**: String identifying a frontend preview component to load alongside the form.

### What the User Sees

A modal dialog with:

- A title (from `strings.json` `step.<step_id>.title`)
- An optional description paragraph (from `step.<step_id>.description`)
- Labeled form fields with optional field-level descriptions
- Error messages (red text) on individual fields or at the top
- A "Submit" or "Next" button (controlled by `last_step`)

---

## All 40 Selectors (UI Input Types)

Selectors define the UI widget rendered for each form field. They are used in `data_schema` like:

```python
vol.Required("my_entity"): EntitySelector(
    EntitySelectorConfig(domain="sensor", multiple=True)
)
```

### Complete Selector Reference

| # | Selector | Type String | Purpose | Key Config Options |
|---|---|---|---|---|
| 1 | `ActionSelector` | `"action"` | Automation action sequence editor | — |
| 2 | `AppSelector` | `"app"` | HA OS installed apps picker | `name`, `slug` |
| 3 | `AddonSelector` | `"addon"` | HA supervisor add-on picker | `name`, `slug` |
| 4 | `AreaSelector` | `"area"` | Area picker from area registry | `multiple`, `device` filter, `entity` filter |
| 5 | `AssistPipelineSelector` | `"assist_pipeline"` | Voice assistant pipeline picker | — |
| 6 | `AttributeSelector` | `"attribute"` | Entity attribute picker | `entity_id` (required), `hide_attributes` |
| 7 | `BackupLocationSelector` | `"backup_location"` | Backup destination picker (HA OS) | — |
| 8 | `BooleanSelector` | `"boolean"` | Toggle switch (on/off) | — |
| 9 | `ChooseSelector` | `"choose"` | Conditional input: show different fields based on choice | `choices` (required), `translation_key` |
| 10 | `ColorRGBSelector` | `"color_rgb"` | Color picker wheel | — |
| 11 | `ColorTempSelector` | `"color_temp"` | Color temperature slider | `unit` (mired/kelvin), `min`, `max` |
| 12 | `ConditionSelector` | `"condition"` | Automation condition editor | — |
| 13 | `ConfigEntrySelector` | `"config_entry"` | Pick an existing config entry | `integration` (optional domain filter) |
| 14 | `ConstantSelector` | `"constant"` | Toggle that returns a fixed value when enabled | `value` (required), `label`, `translation_key` |
| 15 | `ConversationAgentSelector` | `"conversation_agent"` | Pick a conversation/AI agent | `language` |
| 16 | `CountrySelector` | `"country"` | Country dropdown (ISO 3166) | `countries` (filter list), `no_sort` |
| 17 | `DateSelector` | `"date"` | Calendar date picker | — |
| 18 | `DateTimeSelector` | `"datetime"` | Date + time picker | — |
| 19 | `DeviceSelector` | `"device"` | Device picker from device registry | `multiple`, `filter` (integration/manufacturer/model), `entity` filter |
| 20 | `DurationSelector` | `"duration"` | Hours:minutes:seconds input | `enable_day`, `enable_millisecond`, `allow_negative` |
| 21 | `EntitySelector` | `"entity"` | Entity picker from entity registry | `multiple`, `reorder`, `filter` (domain/integration/device_class), `include_entities`, `exclude_entities` |
| 22 | `FileSelector` | `"file"` | File upload input | `accept` (required, MIME types e.g. `"image/*"`) |
| 23 | `FloorSelector` | `"floor"` | Floor picker | `multiple`, `device` filter, `entity` filter |
| 24 | `IconSelector` | `"icon"` | MDI icon picker | `placeholder` (default icon) |
| 25 | `LabelSelector` | `"label"` | HA label picker | `multiple` |
| 26 | `LanguageSelector` | `"language"` | Language dropdown (RFC 5646) | `languages` (filter list), `native_name`, `no_sort` |
| 27 | `LocationSelector` | `"location"` | Map pin selector | `radius` (enable radius circle), `icon` |
| 28 | `MediaSelector` | `"media"` | Media browser/picker | `accept` (MIME filter), `multiple` |
| 29 | `NumberSelector` | `"number"` | Numeric input (box or slider) | `min`, `max`, `step`, `unit_of_measurement`, `mode` ("box"/"slider"), `translation_key` |
| 30 | `ObjectSelector` | `"object"` | Arbitrary YAML/dict input | `fields`, `multiple`, `label_field`, `description_field`, `translation_key` |
| 31 | `QrCodeSelector` | `"qr_code"` | Displays a QR code (output-only) | `data` (required), `scale`, `error_correction_level` |
| 32 | `SelectSelector` | `"select"` | Dropdown or list of options | `options` (required), `multiple`, `custom_value`, `mode` ("list"/"dropdown"), `sort`, `translation_key` |
| 33 | `StateSelector` | `"state"` | Entity state picker | `entity_id`, `hide_states`, `multiple` |
| 34 | `StatisticSelector` | `"statistic"` | Long-term statistic ID picker | `multiple` |
| 35 | `TargetSelector` | `"target"` | Entity/device/area target (like action targets) | `entity` filter, `device` filter |
| 36 | `TemplateSelector` | `"template"` | Jinja2 template text input | — |
| 37 | `TextSelector` | `"text"` | Text input field | `multiline`, `prefix`, `suffix`, `type` (text/email/password/url/tel/search), `autocomplete`, `multiple` |
| 38 | `ThemeSelector` | `"theme"` | HA theme picker | `include_default` |
| 39 | `TimeSelector` | `"time"` | Time-of-day picker | — |
| 40 | `TriggerSelector` | `"trigger"` | Automation trigger editor | — |

### Selector Filtering (Entity/Device/Area)

Entity, device, and area selectors support powerful filtering:

```python
EntitySelector(EntitySelectorConfig(
    domain=["sensor", "binary_sensor"],       # One or multiple domains
    device_class="temperature",               # Device class filter
    integration="mqtt",                       # Integration filter
    supported_features=4,                     # Feature bitmask
    multiple=True,                            # Allow multi-select
    reorder=True,                             # Allow drag reorder (requires multiple)
    exclude_entities=["sensor.excluded"],      # Exclude specific entities
    include_entities=["sensor.specific_one"],  # Only show these entities
))
```

### Read-Only Mode

Any selector can be made read-only:

```python
EntitySelector(EntitySelectorConfig(read_only=True))
```

This displays the value but prevents user modification. Useful in options flows to show immutable settings.

---

## Multi-Step Wizards

Multi-step flows are created by having each step method return the next step's call.

### Pattern

```python
class MyConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            self._host = user_input["host"]
            return await self.async_step_credentials()
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({vol.Required("host"): str}),
            last_step=False,  # Shows "Next" button
        )

    async def async_step_credentials(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(
                title=self._host,
                data={"host": self._host, **user_input},
            )
        return self.async_show_form(
            step_id="credentials",
            data_schema=vol.Schema({
                vol.Required("username"): str,
                vol.Required("password"): str,
            }),
            last_step=True,  # Shows "Submit" button
        )
```

### Navigation Behavior

- **`last_step=True`**: Button reads "Submit"
- **`last_step=False`**: Button reads "Next"
- **`last_step=None`** (default): HA auto-detects based on whether subsequent steps exist
- **Back button**: The HA frontend automatically shows a back button on step 2+. No Python code needed.
- **No explicit "back" handler**: The frontend manages back navigation via its own step history stack. You cannot programmatically control back behavior.

---

## Menu Navigation (`async_show_menu`)

Shows a list of clickable options that navigate to different steps. Each option maps to a step method.

### API

```python
async def async_show_menu(
    self,
    *,
    step_id: str | None = None,
    menu_options: Container[str],  # list of step_id strings or dict
    sort: bool = False,            # alphabetize by label
    description_placeholders: Mapping[str, str] | None = None,
) -> FlowResultT
```

### Usage

```python
async def async_step_init(self, user_input=None):
    return self.async_show_menu(
        step_id="init",
        menu_options=["add_area", "edit_area", "remove_area"],
    )

# Each option maps to an async_step_<option> method:
async def async_step_add_area(self, user_input=None):
    ...
async def async_step_edit_area(self, user_input=None):
    ...
async def async_step_remove_area(self, user_input=None):
    ...
```

### What the User Sees

A dialog with a title, optional description, and a list of clickable menu items (styled as buttons/links). Clicking one navigates to that step.

### Translations (strings.json)

```json
{
  "config": {
    "step": {
      "init": {
        "title": "Configuration",
        "description": "What would you like to do?",
        "menu_options": {
          "add_area": "Add a new area",
          "edit_area": "Edit an existing area",
          "remove_area": "Remove an area"
        }
      }
    }
  }
}
```

### Real-World Example (ZHA)

The ZHA integration uses `async_show_menu` to let users choose between recommended and advanced setup strategies.

---

## Progress / Long-Running Tasks (`async_show_progress`)

For long-running background operations (network scanning, firmware updates, etc.).

### API

```python
async def async_show_progress(
    self,
    *,
    step_id: str | None = None,
    progress_action: str,                     # Translation key for progress text
    description_placeholders: Mapping[str, str] | None = None,
    progress_task: asyncio.Task[Any] | None = None,  # REQUIRED since 2024.8
) -> FlowResultT

async def async_update_progress(self, progress: float) -> None
    # progress is 0.0 to 1.0

async def async_show_progress_done(self, *, next_step_id: str) -> FlowResultT
```

### Pattern

```python
async def async_step_setup(self, user_input=None):
    if not hasattr(self, "_setup_task"):
        self._setup_task = self.hass.async_create_task(
            self._do_setup()
        )

    if not self._setup_task.done():
        return self.async_show_progress(
            step_id="setup",
            progress_action="configuring_device",
            progress_task=self._setup_task,
        )

    try:
        result = self._setup_task.result()
    except Exception:
        return self.async_abort(reason="setup_failed")

    return self.async_show_progress_done(next_step_id="finish")

async def async_step_finish(self, user_input=None):
    return self.async_create_entry(title="Device", data={...})
```

### What the User Sees

A spinner/progress bar with translated text. If `async_update_progress(float)` is called, a percentage progress bar is shown. The dialog auto-advances when the task completes.

### Limitations

- `progress_task` is required since HA 2024.8 (previously optional)
- The task must be an `asyncio.Task` object
- The step method is called repeatedly while the task runs (polling pattern)

---

## External Step Flows (OAuth2)

For flows that redirect users to external websites (OAuth2 authorization, third-party linking).

### API

```python
async def async_external_step(
    self,
    *,
    step_id: str | None = None,
    url: str,                                  # External URL to redirect to
    description_placeholders: Mapping[str, str] | None = None,
) -> FlowResultT

async def async_external_step_done(
    self,
    *,
    next_step_id: str,                         # Step to advance to after external completion
) -> FlowResultT
```

### Flow

1. HA shows a dialog telling the user to visit the external URL
2. User completes authorization on external site
3. External site redirects back to an HA webhook endpoint
4. Webhook calls `async_external_step_done(next_step_id="finish")`
5. HA closes the external window and continues the flow

---

## Collapsible Sections

Groups form fields into collapsible sections. Available since HA 2024.x.

### API

```python
from homeassistant.data_entry_flow import section

data_schema = vol.Schema({
    vol.Required("name"): str,
    vol.Required("host"): str,
    # Collapsible section:
    vol.Required("advanced_options"): section(
        vol.Schema({
            vol.Optional("timeout", default=30): int,
            vol.Optional("retry_count", default=3): int,
        }),
        {"collapsed": True},  # Start collapsed
    ),
})
```

### Parameters

- First argument: a `vol.Schema` with the fields in this section
- Second argument: a dict with `{"collapsed": bool}` (whether section starts collapsed)

### Data Structure

User input nests section data:

```python
{
    "name": "My Device",
    "host": "192.168.1.100",
    "advanced_options": {
        "timeout": 30,
        "retry_count": 3,
    }
}
```

### Translations (strings.json)

```json
{
  "config": {
    "step": {
      "user": {
        "data": {
          "name": "Name",
          "host": "Host"
        },
        "sections": {
          "advanced_options": {
            "name": "Advanced Options",
            "description": "Configure advanced settings",
            "data": {
              "timeout": "Timeout (seconds)",
              "retry_count": "Retry Count"
            },
            "data_description": {
              "timeout": "How long to wait before timing out",
              "retry_count": "Number of retries on failure"
            }
          }
        }
      }
    }
  }
}
```

### Icons (icons.json)

```json
{
  "config": {
    "step": {
      "user": {
        "sections": {
          "advanced_options": "mdi:cog-outline"
        }
      }
    }
  }
}
```

### Limitations

- Only a single level of sections is allowed (no nesting sections within sections)
- The section key in the schema becomes part of the data structure

---

## Suggested Values vs Defaults

Two distinct mechanisms for pre-populating form fields.

### Default Value

```python
vol.Optional("field_name", default="my_default"): str
```

- Pre-fills the field with the value
- If user clears the field and submits, `default` is used as the value
- The value is baked into the schema

### Suggested Value

```python
vol.Optional(
    "field_name",
    description={"suggested_value": "my_suggestion"}
): str
```

- Pre-fills the field with the value
- If user clears the field and submits, the value is `None` / omitted
- Useful for edit forms where you want to show current values but allow clearing

### add_suggested_values_to_schema Helper

For options flows where you want to populate a static schema with existing values:

```python
OPTIONS_SCHEMA = vol.Schema({
    vol.Optional("scan_interval", default=30): int,
    vol.Optional("name"): str,
})

return self.async_show_form(
    step_id="init",
    data_schema=self.add_suggested_values_to_schema(
        OPTIONS_SCHEMA,
        self.config_entry.options,  # Existing values become suggestions
    ),
)
```

This merges each key from the dict into the schema as `description={"suggested_value": ...}`, so the form shows current values but the user can clear them.

---

## Read-Only Fields

Display values that the user cannot modify (useful for showing immutable configuration in options flows).

### Usage

```python
vol.Optional(CONF_ENTITY_ID): EntitySelector(
    EntitySelectorConfig(read_only=True)
)
```

Any selector type supports the `read_only=True` config flag. The field is rendered but grayed out / non-interactive.

---

## Error Handling

### Field-Level Errors

```python
errors = {}
if not valid_host(user_input["host"]):
    errors["host"] = "invalid_host"

return self.async_show_form(
    step_id="user",
    data_schema=schema,
    errors=errors,
)
```

The error appears as red text below the specific field.

### Form-Level Errors (base)

```python
errors["base"] = "cannot_connect"
```

Use the key `"base"` for errors not tied to a specific field. Shown at the top of the form.

### Translations

```json
{
  "config": {
    "error": {
      "invalid_host": "Invalid hostname or IP address",
      "cannot_connect": "Unable to connect to the device"
    }
  }
}
```

---

## Description Placeholders & Translations

### strings.json Structure

```json
{
  "title": "My Integration",
  "config": {
    "flow_title": "Configure {name}",
    "step": {
      "user": {
        "title": "Set up {name}",
        "description": "Enter the details for {name} at {host}",
        "data": {
          "host": "Hostname",
          "port": "Port"
        },
        "data_description": {
          "host": "The IP address or hostname of your device",
          "port": "TCP port number (default: 8080)"
        }
      }
    },
    "error": {
      "cannot_connect": "Failed to connect to {host}",
      "invalid_auth": "Invalid credentials"
    },
    "abort": {
      "already_configured": "This device is already configured",
      "not_supported": "This device is not supported"
    }
  }
}
```

### Placeholder Substitution

```python
return self.async_show_form(
    step_id="user",
    description_placeholders={
        "name": device_name,
        "host": device_host,
    },
)
```

Placeholders in curly braces (`{name}`) in strings.json are replaced with values from `description_placeholders`.

### Title Placeholders

For the config flow title shown in the UI list:

```python
self.context["title_placeholders"] = {"name": "My Device"}
```

Priority for flow title resolution:

1. `title_placeholders` (non-empty) + localized `flow_title` string
2. `name` key from `title_placeholders`
3. Localized integration `title`
4. Manifest `name`
5. Domain name

---

## Specialized Flow Types

| Flow Type | Purpose | Entry Point |
|---|---|---|
| **Config Flow** | Initial setup | `async_step_user()` |
| **Options Flow** | Modify settings post-setup | `async_step_init()` |
| **Reconfigure Flow** | Change connection/host settings | `async_step_reconfigure()` |
| **Reauth Flow** | Fix expired credentials | `async_step_reauth()` |
| **Config Subentry Flow** (2025.3+) | Nested config under parent entry | `async_step_user()` |
| **Discovery Flows** | Auto-discovered devices | `async_step_zeroconf()`, `_dhcp()`, `_mqtt()`, etc. |

### Reconfigure Flows

Allow users to reconfigure an existing config entry without removing and re-adding:

```python
async def async_step_reconfigure(self, user_input=None):
    entry = self._get_reconfigure_entry()

    if user_input is not None:
        return self.async_update_reload_and_abort(
            entry,
            data_updates=user_input,
        )

    return self.async_show_form(
        step_id="reconfigure",
        data_schema=self.add_suggested_values_to_schema(
            vol.Schema({vol.Required("host"): str}),
            entry.data,
        ),
    )
```

A "Reconfigure" button appears on the config entry's settings page.

### Reauth Flows

Triggered when authentication fails (token expired, password changed):

```python
async def async_step_reauth(self, entry_data: Mapping[str, Any]):
    return await self.async_step_reauth_confirm()

async def async_step_reauth_confirm(self, user_input=None):
    if user_input is not None:
        entry = self._get_reauth_entry()
        return self.async_update_reload_and_abort(
            entry,
            data_updates=user_input,
        )
    return self.async_show_form(
        step_id="reauth_confirm",
        data_schema=vol.Schema({vol.Required("password"): str}),
    )
```

A notification banner appears on the integration page with a button to start the reauth flow.

### Options Flows

Allow users to modify mutable settings after initial setup:

```python
# In your ConfigFlow class:
@staticmethod
@callback
def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlowHandler:
    return MyOptionsFlow()

class MyOptionsFlow(OptionsFlow):
    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_SCHEMA,
                self.config_entry.options,
            ),
        )
```

Key differences from config flows:

- Entry point is `async_step_init` (not `async_step_user`)
- Access current entry via `self.config_entry`
- `async_create_entry(data=...)` updates `config_entry.options` (does not create a new entry)
- Same UI capabilities: forms, menus, progress, sections, all selectors

### Auto-Reload (OptionsFlowWithReload)

```python
from homeassistant.config_entries import OptionsFlowWithReload

class MyOptionsFlow(OptionsFlowWithReload):
    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        ...
```

Automatically reloads the integration when options change.

---

## Config Subentry Flows (HA 2025.3+)

A newer feature for nested configuration under a parent config entry. Designed for integrations that need multiple configurations sharing one connection/authentication.

### Use Cases

- Multiple AI conversation agents sharing one API key (OpenAI)
- Multiple weather locations sharing one API subscription
- Multiple MQTT entity definitions sharing one broker connection
- Multiple notification targets sharing one gateway

### Data Hierarchy

```
Config Entry (authentication, connection)
  -> Config Subentry (individual configuration, typed)
       -> Device Registry Entry
            -> Entity
```

### Implementation

```python
class MyConfigFlow(ConfigFlow, domain=DOMAIN):
    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {"location": LocationSubentryFlowHandler}

class LocationSubentryFlowHandler(ConfigSubentryFlow):
    async def async_step_user(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(
                title="My Location",
                data=user_input,
            )
        return self.async_show_form(
            step_id="user",
            data_schema=LOCATION_SCHEMA,
        )

    async def async_step_reconfigure(self, user_input=None):
        subentry = self._get_reconfigure_subentry()
        if user_input is not None:
            return self.async_update_and_abort(
                self._get_entry(),
                subentry,
                data=user_input,
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                LOCATION_SCHEMA, subentry.data
            ),
        )
```

### Key APIs

- `self._get_entry()` — Access the parent config entry
- `self._get_reconfigure_subentry()` — Access the subentry being reconfigured

### Translations (strings.json)

```json
{
  "config_subentries": {
    "location": {
      "title": "Weather Location",
      "step": {
        "user": {
          "title": "Add Location",
          "data": {"city": "City"}
        }
      }
    }
  }
}
```

### Limitations

- Only `user` and `reconfigure` steps are supported (no discovery, reauth, import)
- Unique IDs are scoped to the parent entry (not globally unique)
- Deleting a subentry cascades to associated devices and entities
- Does not apply to automations or scripts

---

## Schema-Based Flow Helper (Declarative)

For simple integrations, `SchemaConfigFlowHandler` provides a declarative approach without writing individual step methods.

### Classes

```python
from homeassistant.helpers.schema_config_entry_flow import (
    SchemaConfigFlowHandler,
    SchemaFlowFormStep,
    SchemaFlowMenuStep,
    SchemaFlowError,
)
```

### Usage

```python
CONFIG_FLOW = {
    "user": SchemaFlowFormStep(
        schema=vol.Schema({vol.Required("name"): str}),
        next_step="options",
    ),
    "options": SchemaFlowFormStep(
        schema=vol.Schema({vol.Optional("interval", default=30): int}),
        validate_user_input=validate_input,
    ),
}

OPTIONS_FLOW = {
    "init": SchemaFlowMenuStep(options=["general", "advanced"]),
    "general": SchemaFlowFormStep(schema=GENERAL_SCHEMA),
    "advanced": SchemaFlowFormStep(schema=ADVANCED_SCHEMA),
}

class MyConfigFlow(SchemaConfigFlowHandler, domain=DOMAIN):
    config_flow = CONFIG_FLOW
    options_flow = OPTIONS_FLOW
    options_flow_reloads = True     # Auto-reload on options change

    def async_config_entry_title(self, options):
        return options["name"]
```

### SchemaFlowFormStep Fields

```python
@dataclass
class SchemaFlowFormStep:
    schema: vol.Schema | Callable | None = None
    validate_user_input: Callable | None = None    # Async validator
    next_step: str | Callable | None = None         # Next step or callable returning step name
    suggested_values: Callable | None | UNDEFINED = UNDEFINED
    preview: str | None = None
    description_placeholders: Callable | UNDEFINED = UNDEFINED
```

### Validation with SchemaFlowError

```python
async def validate_input(handler, user_input):
    if user_input["interval"] < 5:
        raise SchemaFlowError("interval_too_small")
    return user_input
```

### Limitation

All input is saved to `config_entry.options`, not `config_entry.data`. This makes `SchemaConfigFlowHandler` unsuitable for storing credentials or connection details.

---

## Preview Support

Allows displaying a live preview alongside the form that updates as the user changes inputs.

### Backend

```python
return self.async_show_form(
    step_id="user",
    data_schema=schema,
    preview="my_preview",  # Name of the preview component
)
```

### Frontend

The frontend dynamically loads a component at `./previews/flow-preview-my_preview`. This is primarily used by core integrations. Custom integrations would need custom frontend components.

---

## Browser Autofill

Two approaches for enabling browser password managers to auto-fill credentials.

### Voluptuous Key Names (Limited)

The frontend automatically maps these specific key names:

- `"username"` -> HTML `autocomplete="username"`
- `"password"` -> HTML `autocomplete="current-password"`

### TextSelector with autocomplete (Full Control)

```python
from homeassistant.helpers.selector import (
    TextSelector, TextSelectorConfig, TextSelectorType
)

vol.Schema({
    vol.Required("username"): TextSelector(
        TextSelectorConfig(
            type=TextSelectorType.EMAIL,
            autocomplete="username",
        )
    ),
    vol.Required("password"): TextSelector(
        TextSelectorConfig(
            type=TextSelectorType.PASSWORD,
            autocomplete="current-password",
        )
    ),
})
```

---

## Icons

### Integration Icon

Place a `icon.png` (256x256) or `icon@2x.png` in the integration directory.

### Section Icons (icons.json)

```json
{
  "config": {
    "step": {
      "user": {
        "sections": {
          "advanced_options": "mdi:cog-outline",
          "network_settings": "mdi:network"
        }
      }
    }
  }
}
```

Icons appear next to the collapsible section headers.

---

## Discovery Flows

Special step methods for auto-discovery. When a device is discovered, HA calls the appropriate step method directly.

### Supported Discovery Sources

| Step Method | Discovery Source | Manifest Key |
|---|---|---|
| `async_step_dhcp` | DHCP discovery | `dhcp` |
| `async_step_bluetooth` | Bluetooth discovery | `bluetooth` |
| `async_step_homekit` | HomeKit discovery | `homekit` |
| `async_step_mqtt` | MQTT discovery | `mqtt` |
| `async_step_ssdp` | SSDP/UPnP discovery | `ssdp` |
| `async_step_zeroconf` | mDNS/Zeroconf | `zeroconf` |
| `async_step_usb` | USB device discovery | `usb` |
| `async_step_hassio` | Supervisor add-on | `hassio` |

### Unique ID Management

```python
async def async_step_zeroconf(self, discovery_info):
    await self.async_set_unique_id(discovery_info.serial)
    self._abort_if_unique_id_configured()  # Abort if already set up
    # Show confirmation form...
```

---

## Entry Lifecycle Helpers

### async_update_reload_and_abort

Used in reconfigure and reauth flows to update an entry and reload:

```python
return self.async_update_reload_and_abort(
    self._get_reconfigure_entry(),
    data_updates={"host": new_host},         # Merge into entry.data
    options_updates={"interval": 60},        # Merge into entry.options
)
```

### Unique ID Helpers

```python
await self.async_set_unique_id("unique_device_id")
self._abort_if_unique_id_configured()         # Abort if ID exists
self._abort_if_unique_id_configured(
    updates={"host": new_host}                # Update existing entry data
)
self._abort_if_unique_id_mismatch()           # For reauth/reconfigure safety
```

### Matching Flows

```python
if hass.config_entries.flow.async_has_matching_flow(self):
    return self.async_abort(reason="already_in_progress")

def is_matching(self, other_flow: Self) -> bool:
    return other_flow.context.get("unique_id") == self.context.get("unique_id")
```

---

## Summary of All UI Patterns

| Pattern | Method/API | What User Sees |
|---|---|---|
| **Form** | `async_show_form()` | Input fields dialog with Submit/Next |
| **Menu** | `async_show_menu()` | Clickable list of navigation options |
| **Progress** | `async_show_progress()` | Spinner/progress bar during background work |
| **External** | `async_external_step()` | Redirect to external site (OAuth) |
| **Abort** | `async_abort()` | Error/info message, flow ends |
| **Create Entry** | `async_create_entry()` | Success message, flow ends |
| **Collapsible Sections** | `section()` in schema | Expandable/collapsible field groups |
| **Read-Only Fields** | `SelectorConfig(read_only=True)` | Grayed-out display-only fields |
| **QR Code Display** | `QrCodeSelector` | Non-interactive QR code display |
| **Multi-Select** | `Selector(multiple=True)` | Tags/chips multi-select |
| **Drag Reorder** | `EntitySelector(reorder=True)` | Reorderable entity list |
| **Map/Location** | `LocationSelector(radius=True)` | Interactive map with pin and radius |
| **Color Picker** | `ColorRGBSelector` | Color wheel |
| **Slider** | `NumberSelector(mode="slider")` | Horizontal slider |
| **Dropdown** | `SelectSelector(mode="dropdown")` | Dropdown menu |
| **Radio List** | `SelectSelector(mode="list")` | Vertically listed radio buttons |
| **Custom Value** | `SelectSelector(custom_value=True)` | Dropdown with free-text input |
| **File Upload** | `FileSelector(accept="image/*")` | File upload button |
| **Template Editor** | `TemplateSelector` | Jinja2 code editor |
| **Icon Picker** | `IconSelector` | MDI icon browser |
| **Duration Input** | `DurationSelector` | H:M:S multi-field input |
| **Date/Time Pickers** | `DateSelector`, `TimeSelector`, `DateTimeSelector` | Calendar/clock pickers |
| **Conditional Fields** | `ChooseSelector` | Different fields based on choice |
| **Password Input** | `TextSelector(type=PASSWORD)` | Masked text input |
| **Multiline Text** | `TextSelector(multiline=True)` | Textarea |
| **Preview** | `preview="name"` on form | Live preview panel next to form |

---

## Relevance to Area Occupancy Detection

Given this integration's multi-area architecture with complex per-area configuration, the most relevant patterns are:

1. **Menu flow** for the options flow init (add/edit/remove area)
2. **Multi-step wizard** for area configuration (sensors -> weights -> thresholds -> decay)
3. **Collapsible sections** for grouping related settings (sensor selection, weights, advanced)
4. **EntitySelector with filters** (`domain=["binary_sensor"]`, `device_class="motion"`, `multiple=True`, `reorder=True`)
5. **NumberSelector with slider mode** for weights and thresholds
6. **SelectSelector** for room purpose/type
7. **BooleanSelector** for feature toggles (Wasp in Box, decay enabled)
8. **Config Subentry Flows** (2025.3+) — the ideal architecture for multi-area: parent entry holds global config, each area is a subentry with its own device

---

## Sources

- [Data Entry Flow | HA Developer Docs](https://developers.home-assistant.io/docs/data_entry_flow_index/)
- [Config Flow | HA Developer Docs](https://developers.home-assistant.io/docs/config_entries_config_flow_handler/)
- [Options Flow | HA Developer Docs](https://developers.home-assistant.io/docs/config_entries_options_flow_handler/)
- [Selectors | HA Docs](https://www.home-assistant.io/docs/blueprint/selectors/)
- [data_entry_flow.py source (HA Core)](https://github.com/home-assistant/core/blob/dev/homeassistant/data_entry_flow.py)
- [selector.py source (HA Core)](https://github.com/home-assistant/core/blob/dev/homeassistant/helpers/selector.py)
- [schema_config_entry_flow.py source (HA Core)](https://github.com/home-assistant/core/blob/dev/homeassistant/helpers/schema_config_entry_flow.py)
- [Config Subentries Architecture Discussion](https://github.com/home-assistant/architecture/discussions/1070)
- [ConfigSubentryFlow Changes Blog](https://developers.home-assistant.io/blog/2025/03/24/config-subentry-flow-changes/)

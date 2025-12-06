# Installation

## HACS

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Hankanman&repository=Area-Occupancy-Detection&category=integration)

1. **Ensure HACS is installed:** If you don't have the [Home Assistant Community Store (HACS)](https://hacs.xyz/) installed, follow their instructions to set it up first.
2. **Navigate to HACS:** Open your Home Assistant frontend and go to HACS in the sidebar.
3. **Search for Area Occupancy Detection:** Search for "Area Occupancy Detection" and select then **Download**.
4. **Restart Home Assistant:** After the download is complete, restart your Home Assistant instance

## Initial Setup

1. Go to **Settings** > **Devices & Services** > **Integrations** > **+ Add Integration**.
2. Search for **Area Occupancy Detection** and select it.
3. **Configure Area Name:**
   - Select a Home Assistant area for this occupancy detection. The area name will be automatically used for the device and entities.
   - You may need to create the area in Home Assistant first if it doesn't exist.
4. **Configure Area Purpose:**
   - Choose the purpose of the area. This sets a sensible default for the decay half-life used when probability decreases. The purpose selection affects how quickly the system "forgets" about occupancy after activity stops.
5. **Configure Sensors:**
   - Select the sensors that will be used to detect occupancy.
   - You will need to select at least one motion/presence sensor for the integration to work.
   - You can then add sensors of many different types to the area to improve the accuracy of the occupancy detection.

## Configuration

When you first create the integration you will be taken straight to configuring the first area.

When adding new areas you will need to navigate to **Integrations** -> **Area Occupancy Detection** -> **Configure (⚙️ Cog icon)**. This will bring up the configuration menu.

There is detailed documentation on the configuration options here: [Configuration](configuration.md).

### Main Menu

The main menu allows you to modify global settings, add a new area, or manage existing areas.

![Main Menu](../images/config_main_menu.png)

### Global Settings

The global settings menu allows you to modify the global settings for the integration, these are limited for now and will be expanded in the future.

![Global Settings](../images/config_global_settings.png)

### Add New Area

The add new area menu allows you to add a new area to the integration.

![Add New Area](../images/config_new_area.png)

### Manage Areas

The manage areas menu allows you to manage existing areas, you can see a summary of each area and the sensors associated with it. You can then select one of the areas to edit or remove.

![Manage Areas](../images/config_manage_areas.png)

![Manage Area](../images/config_manage_area.png)

## Getting Help

If you encounter issues:

1. Search [GitHub Issues](https://github.com/Hankanman/Area-Occupancy-Detection/issues)
2. Join the [Discussion](https://github.com/Hankanman/Area-Occupancy-Detection/discussions)

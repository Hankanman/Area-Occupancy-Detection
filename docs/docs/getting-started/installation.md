# Installation

There are two main ways to install the Area Occupancy Detection integration for Home Assistant:

## Method 1: HACS (Recommended)

1. **Ensure HACS is installed:** If you don't have the [Home Assistant Community Store (HACS)](https://hacs.xyz/) installed, follow their instructions to set it up first.
2. **Navigate to HACS:** Open your Home Assistant frontend and go to HACS in the sidebar.
3. **Go to Integrations:** Click on "Integrations".
4. **Explore & Add Repositories:** Click the vertical ellipsis (three dots) in the top right corner and select "Custom repositories".
5. **Add Custom Repository:**
    * **Repository:** Enter the URL of this integration's GitHub repository: `https://github.com/hanskohl/Area-Occupancy-Detection`
    * **Category:** Select `Integration`.
    * Click **Add**.
6. **Install Integration:** Back in the main HACS Integrations view, search for "Area Occupancy Detection". Click on it and then click the **Download** button. Select the latest version and confirm.
7. **Restart Home Assistant:** After the download is complete, restart your Home Assistant instance (Configuration -> Settings -> Server Management -> Restart, or use the Developer Tools).

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Hankanman&repository=Area-Occupancy-Detection&category=integration)

## Method 2: Manual Installation

1. **Download the Latest Release:** Go to the [Releases page](https://github.com/hanskohl/Area-Occupancy-Detection/releases) of the GitHub repository and download the `area_occupancy.zip` file from the latest release assets.
2. **Unzip the File:** Extract the contents of the downloaded zip file. You should have a folder named `area_occupancy` containing files like `__init__.py`, `manifest.json`, etc.
3. **Access Home Assistant Configuration Directory:** Connect to the machine running your Home Assistant instance (e.g., via Samba, SSH, or the File editor add-on).
4. **Navigate to `custom_components`:** Inside your main Home Assistant configuration directory (where your `configuration.yaml` file is located), find or create a folder named `custom_components`.
5. **Copy Integration Folder:** Copy the entire extracted `area_occupancy` folder (the one containing `__init__.py`) into the `custom_components` folder.
    Your directory structure should look like this:
    ```
    <config_directory>
        └── custom_components/
            └── area_occupancy/
                ├── __init__.py
                ├── manifest.json
                ├── sensor.py
                └── ... (other integration files)
    ```
6. **Restart Home Assistant:** Restart your Home Assistant instance (Configuration -> Settings -> Server Management -> Restart, or use the Developer Tools).

## Verification

After installation:

1. Restart Home Assistant
2. Go to Settings → Devices & Services
3. Click "+ Add Integration"
4. Search for "Area Occupancy Detection"
   - If it appears, installation was successful
   - If not, check Home Assistant logs for errors

### Getting Help

If you encounter issues:

1. Search [GitHub Issues](https://github.com/Hankanman/Area-Occupancy-Detection/issues)
2. Join the [Discussion](https://github.com/Hankanman/Area-Occupancy-Detection/discussions)

## Next Steps

After restarting, proceed to the [Configuration](configuration.md) guide to set up your first Area Occupancy instance.

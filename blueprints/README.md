# Smart Lighting Control Blueprint - User Guide

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Step 1: Import the Blueprint](#step-1-import-the-blueprint)
  - [Step 2: Create an Automation from the Blueprint](#step-2-create-an-automation-from-the-blueprint)
  - [Step 3: Configure the Inputs](#step-3-configure-the-inputs)
  - [Step 4: Save the Automation](#step-4-save-the-automation)
- [How It Works](#how-it-works)
- [Customization Tips](#customization-tips)
- [Troubleshooting](#troubleshooting)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Advanced Customization](#advanced-customization)
- [Conclusion](#conclusion)

## Introduction

The **Smart Lighting Control** blueprint for Home Assistant allows you to automate your room lighting based on various conditions such as occupancy, ambient light levels, sleep mode, and home occupancy status. This guide will walk you through setting up and customizing the blueprint to suit your needs.

## Features

- **Occupancy Detection**: Automatically turns lights on when motion is detected and turns them off after a specified idle time.
- **Illuminance Monitoring**: Adjusts lighting based on the current light levels in the room to ensure optimal brightness.
- **Sleep Mode Integration**: Dims lights during sleep mode to create a relaxed environment.
- **Home Occupancy Consideration**: Prevents lights from turning on when the house is empty unless guest mode is enabled.
- **Customizable Settings**: Allows you to set default values for brightness levels, transition times, and idle timeouts.
- **Sensor Unavailability Handling**: Uses default values when certain sensors are unavailable.
- **Automation Control**: Enables or disables the entire automation with a single toggle.

## Prerequisites

Before you begin, ensure you have the following:

- Home Assistant installed and running.
- Compatible sensors and devices:
  - Occupancy (motion) sensor.
  - Illuminance (light level) sensor.
  - Input booleans or binary sensors for:
    - Automation enable/disable.
    - People home status.
    - Guest mode.
    - Sleep mode.
  - Lights that can be controlled via Home Assistant.

## Setup Instructions

### Step 1: Import the Blueprint

1. Copy the YAML code of the **Smart Lighting Control** blueprint.
2. In Home Assistant, navigate to **Configuration** > **Automations & Scenes** > **Blueprints**.
3. Click on the **Import Blueprint** button.
4. Paste the YAML code into the text field and click **Preview Import**.
5. Confirm the import by clicking **Import Blueprint**.

### Step 2: Create an Automation from the Blueprint

1. After importing, click on the **Create Automation** button associated with the **Smart Lighting Control** blueprint.
2. You will be presented with a form to fill in the required inputs.

### Step 3: Configure the Inputs

#### 1. Enable Automation

- **Enable Automation**: Select an input boolean or binary sensor to toggle the automation on or off.
  - _Example_: `input_boolean.enable_smart_lighting`

#### 2. Home Occupancy Settings

- **People Home**: Select an entity that indicates if people are home.
  - _Example_: `input_boolean.people_home`
- **Guest Mode**: Select an entity that allows the automation to run even when people are not home.
  - _Example_: `input_boolean.guest_mode`

#### 3. Sleep Mode

- **Sleep Mode**: Select an entity that represents sleep mode.
  - _Example_: `input_boolean.sleep_mode`

#### 4. Area Lights

- **Area Lights**: Choose the lights you want to control with this automation.
  - _Example_: `light.living_room_main`, `light.living_room_floor_lamp`

#### 5. Occupancy Sensor

- **Occupancy Sensor**: Select the binary sensor that detects occupancy.
  - _Example_: `binary_sensor.living_room_motion`

#### 6. Illuminance Sensor

- **Illuminance Sensor**: Select the sensor that measures the room's brightness.
  - _Example_: `sensor.living_room_illuminance`

#### 7. Illuminance Threshold

- **Illuminance Threshold**: Choose an entity that stores your desired illuminance threshold (in lux).
  - _Example_: `input_number.illuminance_threshold`
- **Use Default Illuminance Threshold**: Toggle this on if you want to use a default value when the sensor is unavailable.
- **Default Illuminance Threshold**: Set the default illuminance threshold value (in lux).

#### 8. Brightness Settings

- **Maximum Brightness Sensor**: Select an entity that defines the maximum brightness percentage.
  - _Example_: `input_number.max_brightness`
- **Use Default Maximum Brightness**: Toggle this on to use a default value if the sensor is unavailable.
- **Default Maximum Brightness**: Set the default maximum brightness percentage.

- **Minimum Brightness Sensor**: Select an entity that defines the minimum brightness percentage.
  - _Example_: `input_number.min_brightness`
- **Use Default Minimum Brightness**: Toggle this on to use a default value if the sensor is unavailable.
- **Default Minimum Brightness**: Set the default minimum brightness percentage.

#### 9. Transition Time

- **Transition Time Sensor**: Select an entity that defines the transition time (in seconds) when changing light states.
  - _Example_: `input_number.transition_time`
- **Use Default Transition Time**: Toggle this on to use a default value if the sensor is unavailable.
- **Default Transition Time**: Set the default transition time (in seconds).

#### 10. Idle Timeout

- **Idle Timeout Sensor**: Select an entity that defines the idle timeout duration (in seconds) after which lights turn off when no occupancy is detected.
  - _Example_: `input_number.idle_timeout`
- **Use Default Idle Timeout**: Toggle this on to use a default value if the sensor is unavailable.
- **Default Idle Timeout**: Set the default idle timeout duration (in seconds).

### Step 4: Save the Automation

- After filling in all the required inputs, give your automation a name, such as **Living Room Smart Lighting**.
- Click **Save** to create the automation.

## How It Works

- **Occupancy Detection**: When motion is detected by the occupancy sensor, the automation checks if the current illuminance is below the threshold.

  - If it's dark enough, the lights turn on.
  - If sleep mode is active, lights turn on to the minimum brightness.
  - If not, lights turn on to the maximum brightness.

- **Idle Timeout**: When no motion is detected, the automation waits for the specified idle timeout duration.

  - After the delay, it checks again if the room is still unoccupied.
  - If unoccupied, the lights turn off.

- **Automation Control**: The automation only runs if:

  - The automation is enabled via the **Enable Automation** toggle.
  - Someone is home, or guest mode is enabled.

- **Sensor Unavailability**: If any of the sensors (e.g., illuminance threshold, brightness levels) are unavailable, the automation uses the default values you've set.

## Customization Tips

- **Adjust Brightness Levels**:

  - Use **input_number** entities to easily adjust brightness levels from the UI without editing the automation.
  - Example: Create `input_number.max_brightness` with a range from 0 to 100.

- **Modify Transition Times**:

  - Smooth transitions can enhance the user experience.
  - Adjust the transition time to suit your preference, longer times for gradual changes.

- **Set Appropriate Idle Timeouts**:

  - Shorter timeouts save energy but may turn off lights too quickly.
  - Longer timeouts provide more comfort but may waste energy.

- **Create Multiple Automations**:
  - Use the blueprint to create separate automations for different rooms.
  - Customize settings per room based on usage patterns.

## Troubleshooting

- **Lights Not Turning On**:

  - Check if the automation is enabled.
  - Ensure that the occupancy sensor is detecting motion.
  - Verify that the illuminance sensor is reporting values below the threshold.

- **Lights Not Turning Off**:

  - Ensure the occupancy sensor is correctly reporting the absence of motion.
  - Check the idle timeout duration; it may be set too high.

- **Unexpected Behavior**:
  - Review the entities selected in the automation inputs.
  - Use Home Assistant's **Developer Tools** to monitor sensor states and automation triggers.

## Frequently Asked Questions

**Q:** _Can I use the automation during the day?_

**A:** Yes, as long as the illuminance sensor detects light levels below the threshold, the automation will activate the lights.

---

**Q:** _What happens if the illuminance sensor fails?_

**A:** If the sensor is unavailable, the automation uses the default illuminance threshold you've set, ensuring continuous operation.

---

**Q:** _Can I override the automation manually?_

**A:** Yes, you can turn the lights on or off manually. The automation will resume control based on the next trigger (e.g., occupancy detection).

---

**Q:** _How do I adjust settings without editing the automation?_

**A:** Use **input_number** and **input_boolean** entities for settings like brightness and timeouts. You can adjust these from the Home Assistant UI.

---

**Q:** _Is it possible to add more conditions, like weather or time of day?_

**A:** Yes, you can customize the blueprint by editing the YAML code to include additional conditions.

## Advanced Customization

For users comfortable with YAML and Home Assistant automations, you can further customize the blueprint:

- **Add Time-Based Conditions**: Incorporate conditions to only run the automation during specific times of the day.

- **Integrate with Other Modes**: Add conditions for modes like **Away** or **Vacation**.

- **Enhance Occupancy Detection**: Use multiple occupancy sensors or combine with presence detection.

## Conclusion

The **Smart Lighting Control** blueprint provides a flexible and efficient way to automate your home's lighting based on occupancy and ambient conditions. By following this guide, you can set up the automation to enhance comfort, save energy, and tailor the lighting environment to your preferences.

---

If you have any questions or need further assistance, feel free to reach out to the Home Assistant community or consult the official documentation.

Happy automating!

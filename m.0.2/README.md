# Muscle Sensor Signal Processing System

A Python system for processing signals from muscle sensors on the back of the head, integrating historical data, and building character sets from sensor patterns for world engine control.

## Overview

This system processes muscle sensor signals in real-time and uses them to:
- Collect and filter sensor data with historical context
- Extract meaningful features from sensor patterns
- Control a world engine based on current and historical sensor data
- Build character sets from detected patterns in the latter portions of data streams

## Architecture

```
┌─────────────────────┐
│  Muscle Sensors     │
│  (Back of Head)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Sensor Input       │
│  - Multi-channel    │
│  - Historical data  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stream Processor   │
│  - Filtering        │
│  - Feature extract  │
│  - Pattern detect   │
└──────────┬──────────┘
           │
           ├──────────────────────────┐
           ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│  Sensor Mapper      │    │  Character Builder  │
│  - Surface mapping  │    │  - Pattern mapping  │
│  - World entry map  │    │  - Char generation  │
│  - Entry points     │    └─────────────────────┘
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  World Engine       │
│  - Control signals  │
│  - State management │
│  - Total M world    │
└─────────────────────┘
```

## Components

### 1. Sensor Input (`sensor_input.py`)
- Reads muscle sensor signals from multiple channels
- Maintains historical sensor data
- Supports configurable sampling rates and channels

### 2. Stream Processor (`stream_processor.py`)
- Applies bandpass filtering to remove noise
- Extracts time and frequency domain features
- Detects activation patterns in signals
- Manages continuous data stream buffering

### 3. World Engine (`world_engine.py`)
- Integrates current and historical sensor data
- Generates control signals for world engine
- Maintains world state based on patterns
- Supports aspect-based modifications

### 4. Character Builder (`character_builder.py`)
- Maps sensor patterns to characters
- Builds character sets from feature streams
- Focuses on latter portions of data streams
- Tracks pattern frequencies

### 5. Sensor Mapper (`sensor_mapper.py`)
- Maps total sensor input to 2D surface representations
- Creates world entry maps from muscle sensor data
- Identifies activation regions and entry points
- Visualizes surface topology and world portals

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package:
```bash
pip install -e .
```

## Usage

### Basic Usage

Run the main application:
```bash
python src/main.py
```

This will:
1. Initialize all system components
2. Start sensor acquisition
3. Process 10 iterations of sensor data
4. Display results including:
   - Sensor statistics
   - World engine status
   - Generated character set
   - Character sequence built from patterns

### Using Individual Components

```python
from sensor_input import MuscleSensorInput
from stream_processor import DataStreamProcessor
from world_engine import WorldEngineControl
from character_builder import CharacterSetBuilder
from sensor_mapper import MuscleSensorMapper

# Initialize components
sensor = MuscleSensorInput(num_channels=8, sampling_rate=1000)
processor = DataStreamProcessor(sampling_rate=1000)
world_engine = WorldEngineControl(num_control_channels=4)
char_builder = CharacterSetBuilder()
mapper = MuscleSensorMapper(num_channels=8)

# Start acquisition
sensor.start_acquisition()

# Read sensor data
sensor_data = sensor.read_sensor(duration_ms=100)

# Process data
processor.add_to_stream(sensor_data)
result = processor.process_current_buffer()

# Map input to surface
surface_map = mapper.map_input_to_surface(sensor_data)

# Create world entry map
world_entry_map = mapper.create_world_entry_map(surface_map)

# Generate control signals
control_signals = world_engine.integrate_sensor_data(
    result['features'],
    processor.get_feature_history()
)

# Build characters from patterns
char = char_builder.add_pattern_from_features(result['features'])

# Total input to world
all_sensor_data = sensor.get_historical_data()
world_info = mapper.total_input_to_world(all_sensor_data)
print(f"World entries: {world_info['total_entries']}")
```

## Configuration

### Sensor Configuration
- `num_channels`: Number of sensor channels (default: 8)
- `sampling_rate`: Sampling rate in Hz (default: 1000)

### Processing Configuration
- `buffer_size`: Number of samples to buffer (default: 10)
- `lowcut`/`highcut`: Bandpass filter frequencies (20-450 Hz)

### World Engine Configuration
- `num_control_channels`: Number of control outputs (default: 4)

## Example Output

```
============================================================
Muscle Sensor Signal Processing System
============================================================

Initializing components...
✓ All components initialized

✓ Sensor acquisition started

Running processing loop (10 iterations)...
------------------------------------------------------------
Iteration 1: Generated character 'A' | Patterns: 0 | Control avg: 0.042
Iteration 2: Generated character 'B' | Patterns: 0 | Control avg: -0.018
...
------------------------------------------------------------

============================================================
RESULTS
============================================================

Sensor Statistics:
  Total readings: 10
  Total samples: 1000
  Duration: 1.00 seconds

World Engine Status:
  Active patterns: 0
  Control history length: 10
  Current control signals: [-0.123, 0.456]...

Character Set:
  Total characters: 10
  Unique patterns: 10
  Character sequence: ABCDEFGHIJ

Built Character Sequence (from latter portions):
  FGHIJ

Surface Map Summary:
  Map resolution: (50, 50)
  Entry points: 3
  Mean activation: 0.234
  Total world entries: 3

✓ Visualization saved to surface_map.png
✓ System shutdown complete
============================================================
```

## Features

- **Real-time Processing**: Continuous sensor data acquisition and processing
- **Historical Integration**: Leverages previous data for better control
- **Pattern Recognition**: Detects activation patterns in muscle signals
- **Surface Mapping**: Maps total sensor input to 2D surface representations
- **World Entry Creation**: Creates entry points and portals into world engine
- **Adaptive Control**: Generates control signals based on sensor input
- **Character Generation**: Builds character sets from detected patterns
- **Visualization**: Surface maps and activation region displays
- **Configurable**: Flexible parameters for different use cases

## Dependencies

- `numpy`: Numerical computing
- `scipy`: Signal processing and filtering
- `matplotlib`: Visualization (optional)
- `pandas`: Data management (optional)

## Project Structure

```
m.0.2/
├── .github/
│   └── copilot-instructions.md
├── src/
│   ├── __init__.py
│   ├── sensor_input.py
│   ├── stream_processor.py
│   ├── world_engine.py
│   ├── character_builder.py
│   ├── sensor_mapper.py
│   └── main.py
├── requirements.txt
├── setup.py
└── README.md
```

## Future Enhancements

- Hardware sensor integration
- Machine learning for pattern classification
- Real-time visualization dashboard
- Configurable character mapping schemes
- Multi-user support
- Data logging and replay

## License

This project is provided as-is for educational and research purposes.

## Support

For questions or issues, please refer to the project documentation or create an issue in the repository.

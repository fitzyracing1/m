"""
Muscle Sensor Signal Processing System

Main application integrating all components:
- Sensor input from head muscle sensors
- Data stream processing with historical context
- World engine control
- Character set building from patterns
- Sensor mapping and surface creation
"""

import sys
from sensor_input import MuscleSensorInput
from stream_processor import DataStreamProcessor
from world_engine import WorldEngineControl
from character_builder import CharacterSetBuilder
from sensor_mapper import MuscleSensorMapper


def main():
    """Main application entry point."""
    print("=" * 60)
    print("Muscle Sensor Signal Processing System")
    print("=" * 60)
    print()
    
    # Initialize components
    print("Initializing components...")
    sensor = MuscleSensorInput(num_channels=8, sampling_rate=1000)
    processor = DataStreamProcessor(sampling_rate=1000, buffer_size=10)
    world_engine = WorldEngineControl(num_control_channels=4)
    char_builder = CharacterSetBuilder(feature_dimension=40)
    mapper = MuscleSensorMapper(num_channels=8)
    print("✓ All components initialized")
    print()
    
    # Display sensor topology
    topology = mapper.get_sensor_topology()
    print("Sensor Topology:")
    print(f"  Number of sensors: {topology['num_sensors']}")
    print(f"  Average distance: {topology['avg_distance']:.3f}")
    print()
    
    # Start sensor acquisition
    sensor.start_acquisition()
    print("✓ Sensor acquisition started")
    print()
    
    # Simulate processing loop
    print("Running processing loop (10 iterations)...")
    print("-" * 60)
    
    # Store sensor data for mapping
    all_sensor_data = []
    
    for i in range(10):
        # Read sensor data
        sensor_data = sensor.read_sensor(duration_ms=100)
        all_sensor_data.append(sensor_data)
        
        # Add to processing stream
        processor.add_to_stream(sensor_data)
        
        # Process current buffer
        result = processor.process_current_buffer()
        
        if result["status"] == "processed":
            features = result["features"]
            
            # Get historical features
            historical_features = processor.get_feature_history(n_samples=5)
            
            # Integrate with world engine
            control_signals = world_engine.integrate_sensor_data(
                features, historical_features
            )
            
            # Update world state
            world_engine.update_world_state(control_signals, result)
            
            # Build character from pattern (latter portions)
            if len(historical_features) > 5:
                char = char_builder.add_pattern_from_features(features)
                print(f"Iteration {i+1}: Generated character '{char}' | "
                      f"Patterns: {result['num_patterns']} | "
                      f"Control avg: {sum(control_signals.values())/len(control_signals):.3f}")
    
    print("-" * 60)
    print()
    
    # Create surface maps and world entry
    print("Creating surface maps and world entry...")
    print("-" * 60)
    
    # Map total input to world
    world_entry_info = mapper.total_input_to_world(all_sensor_data)
    
    print(f"✓ Surface mapping complete")
    print(f"  Total readings processed: {world_entry_info['total_readings']}")
    print(f"  Entry points detected: {world_entry_info['num_entry_points']}")
    print(f"  Total world entries: {world_entry_info['total_entries']}")
    print(f"  Max activation: {world_entry_info['max_activation']:.3f}")
    print()
    
    # Display regional summary
    print("Surface Regional Summary:")
    for region, value in world_entry_info['regional_summary'].items():
        print(f"  {region}: {value:.3f}")
    print("-" * 60)
    print()
    
    # Display results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    # Sensor statistics
    sensor_stats = sensor.get_statistics()
    print("Sensor Statistics:")
    print(f"  Total readings: {sensor_stats['total_readings']}")
    print(f"  Total samples: {sensor_stats['total_samples']}")
    print(f"  Duration: {sensor_stats['duration_seconds']:.2f} seconds")
    print()
    
    # World engine status
    world_status = world_engine.get_world_status()
    print("World Engine Status:")
    print(f"  Active patterns: {world_status['active_patterns']}")
    print(f"  Control history length: {world_status['control_history_length']}")
    print(f"  Current control signals: {list(world_status['current_controls'].values())[:2]}...")
    print()
    
    # Character set info
    char_info = char_builder.get_character_set_info()
    print("Character Set:")
    print(f"  Total characters: {char_info['total_characters']}")
    print(f"  Unique patterns: {char_info['unique_patterns']}")
    print(f"  Character sequence: {char_info['character_set']}")
    print()
    
    # Build final character sequence from latter portions
    all_features = processor.get_feature_history()
    final_sequence = char_builder.build_from_feature_stream(all_features)
    print("Built Character Sequence (from latter portions):")
    print(f"  {final_sequence}")
    print()
    
    # Surface Map Summary
    print("Surface Map Summary:")
    print(f"  Map resolution: {world_entry_info['entry_map_shape']}")
    print(f"  Entry points: {world_entry_info['num_entry_points']}")
    print(f"  Mean activation: {world_entry_info['mean_activation']:.3f}")
    print(f"  Total world entries: {world_entry_info['total_entries']}")
    print()
    
    # Try to visualize (optional, won't fail if display unavailable)
    try:
        print("Generating surface map visualization...")
        mapper.visualize_surface_map(save_path='surface_map.png')
        print("✓ Visualization saved to surface_map.png")
    except Exception as e:
        print(f"  (Visualization skipped: {str(e)})")
    print()
    
    # Stop sensor
    sensor.stop_acquisition()
    print("✓ System shutdown complete")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

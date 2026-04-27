"""Sensor input module for reading muscle sensor signals."""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime


class SensorData:
    """Represents a single sensor reading."""
    
    def __init__(self, timestamp: datetime, values: np.ndarray, sensor_id: str = "head_muscle_sensor"):
        self.timestamp = timestamp
        self.values = values
        self.sensor_id = sensor_id
    
    def __repr__(self):
        return f"SensorData(sensor_id={self.sensor_id}, timestamp={self.timestamp}, shape={self.values.shape})"


class MuscleSensorInput:
    """Handles input from muscle sensors on the back of the head."""
    
    def __init__(self, num_channels: int = 8, sampling_rate: int = 1000):
        """
        Initialize the muscle sensor input handler.
        
        Args:
            num_channels: Number of sensor channels
            sampling_rate: Sampling rate in Hz
        """
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.history: List[SensorData] = []
        self.is_active = False
    
    def start_acquisition(self):
        """Start acquiring sensor data."""
        self.is_active = True
        print(f"Sensor acquisition started: {self.num_channels} channels @ {self.sampling_rate}Hz")
    
    def stop_acquisition(self):
        """Stop acquiring sensor data."""
        self.is_active = False
        print("Sensor acquisition stopped")
    
    def read_sensor(self, duration_ms: int = 100) -> SensorData:
        """
        Read sensor data for a specified duration.
        
        Args:
            duration_ms: Duration in milliseconds
            
        Returns:
            SensorData object containing the reading
        """
        num_samples = int(self.sampling_rate * duration_ms / 1000)
        # Simulate sensor reading (in real implementation, this would read from hardware)
        values = np.random.randn(self.num_channels, num_samples) * 0.1
        
        sensor_data = SensorData(
            timestamp=datetime.now(),
            values=values,
            sensor_id="head_muscle_sensor"
        )
        
        self.history.append(sensor_data)
        return sensor_data
    
    def get_historical_data(self, n_samples: Optional[int] = None) -> List[SensorData]:
        """
        Retrieve historical sensor data.
        
        Args:
            n_samples: Number of most recent samples to retrieve (None for all)
            
        Returns:
            List of SensorData objects
        """
        if n_samples is None:
            return self.history
        return self.history[-n_samples:]
    
    def clear_history(self):
        """Clear historical sensor data."""
        self.history.clear()
        print("Sensor history cleared")
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected sensor data."""
        if not self.history:
            return {"total_samples": 0}
        
        total_samples = sum(data.values.shape[1] for data in self.history)
        duration_seconds = total_samples / self.sampling_rate
        
        return {
            "total_readings": len(self.history),
            "total_samples": total_samples,
            "duration_seconds": duration_seconds,
            "num_channels": self.num_channels,
            "sampling_rate": self.sampling_rate
        }

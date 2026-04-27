"""Sensor mapping module for creating surface maps from muscle sensor input."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sensor_input import SensorData


class SensorPosition:
    """Represents the physical position of a sensor."""
    
    def __init__(self, x: float, y: float, z: float, sensor_id: str):
        self.x = x
        self.y = y
        self.z = z
        self.sensor_id = sensor_id
        self.position = np.array([x, y, z])
    
    def distance_to(self, other: 'SensorPosition') -> float:
        """Calculate distance to another sensor position."""
        return np.linalg.norm(self.position - other.position)
    
    def __repr__(self):
        return f"SensorPosition({self.sensor_id}, [{self.x:.2f}, {self.y:.2f}, {self.z:.2f}])"


class SurfaceMap:
    """Represents a surface map created from sensor data."""
    
    def __init__(self, resolution: Tuple[int, int] = (50, 50)):
        self.resolution = resolution
        self.map_data = np.zeros(resolution)
        self.intensity_map = np.zeros(resolution)
        self.activation_regions: List[Tuple[int, int, float]] = []
    
    def update(self, grid_data: np.ndarray):
        """Update the surface map with new grid data."""
        self.map_data = grid_data
        self.intensity_map = np.abs(grid_data)
    
    def find_activation_regions(self, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """Find regions of high activation on the surface."""
        self.activation_regions.clear()
        
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                if self.intensity_map[i, j] > threshold:
                    self.activation_regions.append((i, j, self.intensity_map[i, j]))
        
        return self.activation_regions
    
    def get_regional_summary(self, num_regions: int = 4) -> Dict[str, float]:
        """Get summary statistics for different regions of the map."""
        h, w = self.resolution
        h_half, w_half = h // 2, w // 2
        
        regions = {
            'top_left': self.intensity_map[:h_half, :w_half],
            'top_right': self.intensity_map[:h_half, w_half:],
            'bottom_left': self.intensity_map[h_half:, :w_half],
            'bottom_right': self.intensity_map[h_half:, w_half:]
        }
        
        return {
            name: float(np.mean(region))
            for name, region in regions.items()
        }


class MuscleSensorMapper:
    """Maps muscle sensor input to create surface representations for world entry."""
    
    def __init__(self, num_channels: int = 8):
        """
        Initialize the muscle sensor mapper.
        
        Args:
            num_channels: Number of sensor channels
        """
        self.num_channels = num_channels
        self.sensor_positions = self._initialize_sensor_positions()
        self.surface_map = SurfaceMap(resolution=(50, 50))
        self.world_entry_map = None
        self.total_entries = 0
    
    def _initialize_sensor_positions(self) -> List[SensorPosition]:
        """Initialize sensor positions on the back of the head."""
        positions = []
        
        # Create a grid of sensors on the back of the head
        # Coordinates represent positions on a curved surface
        sensor_layout = [
            # Upper back of head
            (-0.5, 0.8, 0.3, "sensor_0"),
            (0.0, 0.9, 0.4, "sensor_1"),
            (0.5, 0.8, 0.3, "sensor_2"),
            
            # Middle back of head
            (-0.6, 0.5, 0.1, "sensor_3"),
            (0.0, 0.6, 0.2, "sensor_4"),
            (0.6, 0.5, 0.1, "sensor_5"),
            
            # Lower back of head
            (-0.4, 0.2, -0.2, "sensor_6"),
            (0.4, 0.2, -0.2, "sensor_7"),
        ]
        
        for x, y, z, sid in sensor_layout[:self.num_channels]:
            positions.append(SensorPosition(x, y, z, sid))
        
        return positions
    
    def map_input_to_surface(self, sensor_data: SensorData) -> SurfaceMap:
        """
        Map sensor input to a surface representation.
        
        Args:
            sensor_data: Sensor data to map
            
        Returns:
            Surface map representation
        """
        # Average across time dimension
        sensor_values = np.mean(sensor_data.values, axis=1)
        
        # Create 2D grid using interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1, 1, self.surface_map.resolution[0]),
            np.linspace(-1, 1, self.surface_map.resolution[1])
        )
        
        # Interpolate sensor values onto grid
        grid_values = np.zeros(self.surface_map.resolution)
        
        for idx, (pos, value) in enumerate(zip(self.sensor_positions, sensor_values)):
            # Gaussian interpolation from each sensor position
            distances = np.sqrt((grid_x - pos.x)**2 + (grid_y - pos.y)**2)
            weights = np.exp(-distances**2 / 0.2)
            grid_values += weights * value
        
        # Normalize
        grid_values = grid_values / (np.max(np.abs(grid_values)) + 1e-10)
        
        # Update surface map
        self.surface_map.update(grid_values)
        
        return self.surface_map
    
    def create_world_entry_map(self, surface_map: SurfaceMap) -> np.ndarray:
        """
        Create a world entry map from the surface map.
        
        Args:
            surface_map: Surface map to convert
            
        Returns:
            World entry map array
        """
        # Transform surface map to world entry coordinates
        entry_map = np.copy(surface_map.map_data)
        
        # Apply transformations for world entry
        # 1. Enhance high-activation regions
        entry_map = np.tanh(entry_map * 2.0)
        
        # 2. Create entry portals at activation peaks
        peaks = surface_map.find_activation_regions(threshold=0.6)
        
        for i, j, intensity in peaks:
            # Create entry portal effect
            radius = 3
            for di in range(-radius, radius+1):
                for dj in range(-radius, radius+1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < entry_map.shape[0] and 0 <= nj < entry_map.shape[1]:
                        dist = np.sqrt(di**2 + dj**2)
                        if dist <= radius:
                            entry_map[ni, nj] = max(entry_map[ni, nj], intensity * (1 - dist/radius))
        
        self.world_entry_map = entry_map
        self.total_entries += len(peaks)
        
        return entry_map
    
    def total_input_to_world(self, sensor_data_list: List[SensorData]) -> Dict:
        """
        Process total sensor input and create complete world entry map.
        
        Args:
            sensor_data_list: List of sensor data readings
            
        Returns:
            Dictionary with world entry information
        """
        if not sensor_data_list:
            return {"status": "no_data"}
        
        # Aggregate all sensor data
        all_surface_maps = []
        
        for sensor_data in sensor_data_list:
            surface_map = self.map_input_to_surface(sensor_data)
            all_surface_maps.append(surface_map.map_data.copy())
        
        # Create combined surface map
        combined_surface = np.mean(all_surface_maps, axis=0)
        self.surface_map.update(combined_surface)
        
        # Create world entry map
        entry_map = self.create_world_entry_map(self.surface_map)
        
        # Get regional information
        regional_summary = self.surface_map.get_regional_summary()
        
        # Find entry points
        activation_regions = self.surface_map.find_activation_regions(threshold=0.5)
        
        return {
            "status": "complete",
            "total_sensors": self.num_channels,
            "total_readings": len(sensor_data_list),
            "entry_map_shape": entry_map.shape,
            "num_entry_points": len(activation_regions),
            "regional_summary": regional_summary,
            "total_entries": self.total_entries,
            "max_activation": float(np.max(entry_map)),
            "mean_activation": float(np.mean(entry_map))
        }
    
    def visualize_surface_map(self, save_path: Optional[str] = None):
        """
        Visualize the surface map.
        
        Args:
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Surface map
        im1 = axes[0].imshow(self.surface_map.intensity_map, cmap='hot', origin='lower')
        axes[0].set_title('Muscle Sensor Surface Map')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        plt.colorbar(im1, ax=axes[0], label='Intensity')
        
        # Mark sensor positions
        for pos in self.sensor_positions:
            x_idx = int((pos.x + 1) * self.surface_map.resolution[0] / 2)
            y_idx = int((pos.y + 1) * self.surface_map.resolution[1] / 2)
            axes[0].plot(x_idx, y_idx, 'b*', markersize=10)
        
        # World entry map
        if self.world_entry_map is not None:
            im2 = axes[1].imshow(self.world_entry_map, cmap='viridis', origin='lower')
            axes[1].set_title('World Entry Map')
            axes[1].set_xlabel('X Position')
            axes[1].set_ylabel('Y Position')
            plt.colorbar(im2, ax=axes[1], label='Entry Intensity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def get_sensor_topology(self) -> Dict:
        """Get information about sensor topology."""
        distances = []
        for i, pos1 in enumerate(self.sensor_positions):
            for j, pos2 in enumerate(self.sensor_positions[i+1:], i+1):
                distances.append(pos1.distance_to(pos2))
        
        return {
            "num_sensors": len(self.sensor_positions),
            "avg_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "positions": [str(pos) for pos in self.sensor_positions]
        }

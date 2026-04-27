"""World engine integration for sensor data stream control."""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


class WorldState:
    """Represents the current state of the world engine."""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.parameters: Dict[str, Any] = {}
        self.active_patterns: List[str] = []
        self.control_signals: Dict[str, float] = {}
    
    def update(self, key: str, value: Any):
        """Update a world state parameter."""
        self.parameters[key] = value
        self.timestamp = datetime.now()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a world state parameter."""
        return self.parameters.get(key, default)
    
    def __repr__(self):
        return f"WorldState(timestamp={self.timestamp}, params={len(self.parameters)})"


class WorldEngineControl:
    """Controls world engine based on sensor input and historical data."""
    
    def __init__(self, num_control_channels: int = 4):
        """
        Initialize the world engine control.
        
        Args:
            num_control_channels: Number of control channels for the world engine
        """
        self.num_control_channels = num_control_channels
        self.world_state = WorldState()
        self.control_history: List[Dict[str, float]] = []
        self.mapping_matrix = self._initialize_mapping()
    
    def _initialize_mapping(self) -> np.ndarray:
        """Initialize the feature-to-control mapping matrix."""
        # Creates a mapping from sensor features to world control signals
        # In a real implementation, this could be learned or configured
        return np.random.randn(self.num_control_channels, 40) * 0.1
    
    def integrate_sensor_data(self, features: np.ndarray, historical_features: List[np.ndarray]) -> Dict[str, float]:
        """
        Integrate current and historical sensor data to generate control signals.
        
        Args:
            features: Current feature vector
            historical_features: List of historical feature vectors
            
        Returns:
            Dictionary of control signals
        """
        # Combine current and historical data with weighted average
        if historical_features:
            # Weight recent history more heavily
            weights = np.exp(-np.arange(len(historical_features))[::-1] * 0.1)
            weights = weights / np.sum(weights)
            
            historical_avg = np.average(
                np.array(historical_features),
                axis=0,
                weights=weights
            )
            
            # Blend current with historical (70% current, 30% historical)
            blended_features = 0.7 * features + 0.3 * historical_avg
        else:
            blended_features = features
        
        # Pad or truncate features to match mapping matrix
        if len(blended_features) < self.mapping_matrix.shape[1]:
            padded_features = np.zeros(self.mapping_matrix.shape[1])
            padded_features[:len(blended_features)] = blended_features
            blended_features = padded_features
        else:
            blended_features = blended_features[:self.mapping_matrix.shape[1]]
        
        # Generate control signals
        control_signals = np.tanh(self.mapping_matrix @ blended_features)
        
        # Convert to dictionary
        control_dict = {
            f"control_channel_{i}": float(signal)
            for i, signal in enumerate(control_signals)
        }
        
        self.control_history.append(control_dict)
        return control_dict
    
    def update_world_state(self, control_signals: Dict[str, float], pattern_info: Dict):
        """
        Update the world state based on control signals and detected patterns.
        
        Args:
            control_signals: Dictionary of control channel values
            pattern_info: Information about detected patterns
        """
        # Update control signals in world state
        self.world_state.control_signals = control_signals
        
        # Update based on pattern detection
        if pattern_info.get("num_patterns", 0) > 0:
            self.world_state.active_patterns.append(
                f"pattern_{len(self.world_state.active_patterns)}"
            )
        
        # Update other world parameters
        self.world_state.update("num_patterns_detected", pattern_info.get("num_patterns", 0))
        self.world_state.update("last_update", datetime.now())
    
    def get_control_stream(self) -> List[Dict[str, float]]:
        """
        Get the full control signal history.
        
        Returns:
            List of control signal dictionaries
        """
        return self.control_history
    
    def apply_aspect_modification(self, aspect_factor: float = 1.0):
        """
        Apply aspect-based modifications to the mapping.
        
        Args:
            aspect_factor: Factor for aspect-based scaling
        """
        self.mapping_matrix *= aspect_factor
        print(f"Applied aspect modification with factor: {aspect_factor}")
    
    def get_world_status(self) -> Dict:
        """
        Get current world engine status.
        
        Returns:
            Dictionary with world engine status information
        """
        return {
            "timestamp": self.world_state.timestamp,
            "num_control_channels": self.num_control_channels,
            "active_patterns": len(self.world_state.active_patterns),
            "control_history_length": len(self.control_history),
            "current_controls": self.world_state.control_signals
        }
    
    def reset_world_state(self):
        """Reset the world state to initial conditions."""
        self.world_state = WorldState()
        self.control_history.clear()
        print("World state reset")

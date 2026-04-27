"""Data stream processor for muscle sensor signals."""

import numpy as np
from typing import List, Tuple, Optional
from scipy import signal
from sensor_input import SensorData


class SignalProcessor:
    """Processes raw sensor signals with filtering and feature extraction."""
    
    def __init__(self, sampling_rate: int = 1000):
        """
        Initialize the signal processor.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.filter_cutoff = 50  # Hz
        self.filter_order = 4
    
    def bandpass_filter(self, data: np.ndarray, lowcut: float = 20, highcut: float = 450) -> np.ndarray:
        """
        Apply bandpass filter to remove noise.
        
        Args:
            data: Input signal array
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            
        Returns:
            Filtered signal
        """
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(self.filter_order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=-1)
        
        return filtered
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from sensor data.
        
        Args:
            data: Input signal array (channels x samples)
            
        Returns:
            Feature vector
        """
        features = []
        
        for channel in data:
            # Time domain features
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            rms = np.sqrt(np.mean(channel**2))
            
            # Frequency domain features
            freqs, psd = signal.welch(channel, self.sampling_rate)
            dominant_freq = freqs[np.argmax(psd)]
            mean_freq = np.sum(freqs * psd) / np.sum(psd)
            
            features.extend([mean_val, std_val, rms, dominant_freq, mean_freq])
        
        return np.array(features)
    
    def detect_patterns(self, data: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """
        Detect activation patterns in the signal.
        
        Args:
            data: Input signal array
            threshold: Activation threshold
            
        Returns:
            List of (start, end) indices for detected patterns
        """
        # Calculate envelope
        envelope = np.abs(signal.hilbert(data))
        envelope = signal.filtfilt(*signal.butter(2, 10, fs=self.sampling_rate), envelope, axis=-1)
        
        # Detect threshold crossings
        patterns = []
        for channel_envelope in envelope:
            above_threshold = channel_envelope > threshold
            crossings = np.diff(above_threshold.astype(int))
            
            starts = np.where(crossings == 1)[0]
            ends = np.where(crossings == -1)[0]
            
            # Ensure pairs
            if len(starts) > 0 and len(ends) > 0:
                if starts[0] > ends[0]:
                    ends = ends[1:]
                if len(starts) > len(ends):
                    starts = starts[:len(ends)]
                
                patterns.extend(zip(starts, ends))
        
        return patterns


class DataStreamProcessor:
    """Manages continuous processing of sensor data streams."""
    
    def __init__(self, sampling_rate: int = 1000, buffer_size: int = 10):
        """
        Initialize the data stream processor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            buffer_size: Number of samples to keep in buffer
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.signal_processor = SignalProcessor(sampling_rate)
        self.data_buffer: List[SensorData] = []
        self.feature_history: List[np.ndarray] = []
    
    def add_to_stream(self, sensor_data: SensorData):
        """
        Add new sensor data to the processing stream.
        
        Args:
            sensor_data: New sensor data to process
        """
        # Add to buffer
        self.data_buffer.append(sensor_data)
        
        # Maintain buffer size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
    
    def process_current_buffer(self) -> dict:
        """
        Process all data in the current buffer.
        
        Returns:
            Dictionary containing processed results
        """
        if not self.data_buffer:
            return {"status": "empty_buffer"}
        
        # Concatenate all buffer data
        all_data = np.concatenate([d.values for d in self.data_buffer], axis=1)
        
        # Apply filtering
        filtered_data = self.signal_processor.bandpass_filter(all_data)
        
        # Extract features
        features = self.signal_processor.extract_features(filtered_data)
        self.feature_history.append(features)
        
        # Detect patterns
        patterns = self.signal_processor.detect_patterns(filtered_data)
        
        return {
            "status": "processed",
            "buffer_size": len(self.data_buffer),
            "features": features,
            "num_patterns": len(patterns),
            "patterns": patterns,
            "data_shape": filtered_data.shape
        }
    
    def get_feature_history(self, n_samples: Optional[int] = None) -> List[np.ndarray]:
        """
        Get historical feature vectors.
        
        Args:
            n_samples: Number of recent samples (None for all)
            
        Returns:
            List of feature vectors
        """
        if n_samples is None:
            return self.feature_history
        return self.feature_history[-n_samples:]
    
    def clear_buffer(self):
        """Clear the data buffer."""
        self.data_buffer.clear()
        print("Stream buffer cleared")

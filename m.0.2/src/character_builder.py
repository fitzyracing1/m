"""Character set builder from sensor patterns."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import string


class PatternCharacterMapping:
    """Maps sensor patterns to characters."""
    
    def __init__(self):
        self.pattern_to_char: Dict[str, str] = {}
        self.char_to_pattern: Dict[str, str] = {}
        self.pattern_frequency: Dict[str, int] = {}
    
    def add_mapping(self, pattern_signature: str, character: str):
        """Add a pattern-to-character mapping."""
        self.pattern_to_char[pattern_signature] = character
        self.char_to_pattern[character] = pattern_signature
        self.pattern_frequency[pattern_signature] = self.pattern_frequency.get(pattern_signature, 0) + 1
    
    def get_character(self, pattern_signature: str) -> Optional[str]:
        """Get the character for a given pattern."""
        return self.pattern_to_char.get(pattern_signature)
    
    def get_pattern(self, character: str) -> Optional[str]:
        """Get the pattern for a given character."""
        return self.char_to_pattern.get(character)
    
    def get_most_frequent_patterns(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the most frequent patterns."""
        sorted_patterns = sorted(
            self.pattern_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_patterns[:n]


class CharacterSetBuilder:
    """Builds character sets from sensor input patterns."""
    
    def __init__(self, feature_dimension: int = 40):
        """
        Initialize the character set builder.
        
        Args:
            feature_dimension: Dimension of feature vectors
        """
        self.feature_dimension = feature_dimension
        self.mapping = PatternCharacterMapping()
        self.pattern_templates: Dict[str, np.ndarray] = {}
        self.character_set: List[str] = []
        self.available_chars = list(string.ascii_uppercase + string.digits + " .,!?")
        self.next_char_index = 0
    
    def _create_pattern_signature(self, features: np.ndarray) -> str:
        """
        Create a signature string from feature vector.
        
        Args:
            features: Feature vector
            
        Returns:
            Pattern signature string
        """
        # Quantize features to create discrete signature
        quantized = np.round(features * 10).astype(int)
        
        # Create signature from most significant features
        top_indices = np.argsort(np.abs(quantized))[-8:]
        signature = "_".join([f"{i}:{quantized[i]}" for i in sorted(top_indices)])
        
        return signature
    
    def add_pattern_from_features(self, features: np.ndarray) -> str:
        """
        Add a new pattern from feature vector and assign a character.
        
        Args:
            features: Feature vector from sensor data
            
        Returns:
            The assigned character
        """
        signature = self._create_pattern_signature(features)
        
        # Check if pattern already exists
        existing_char = self.mapping.get_character(signature)
        if existing_char:
            self.mapping.pattern_frequency[signature] += 1
            return existing_char
        
        # Assign new character
        if self.next_char_index < len(self.available_chars):
            new_char = self.available_chars[self.next_char_index]
            self.next_char_index += 1
        else:
            # Wrap around if we run out of characters
            new_char = f"[{self.next_char_index - len(self.available_chars)}]"
            self.next_char_index += 1
        
        # Store mapping and template
        self.mapping.add_mapping(signature, new_char)
        self.pattern_templates[new_char] = features.copy()
        self.character_set.append(new_char)
        
        return new_char
    
    def build_from_feature_stream(self, feature_history: List[np.ndarray]) -> str:
        """
        Build character sequence from feature history.
        
        Args:
            feature_history: List of feature vectors
            
        Returns:
            String built from patterns
        """
        if not feature_history:
            return ""
        
        # Process later portions of the stream (last 50%)
        latter_portion_start = len(feature_history) // 2
        latter_features = feature_history[latter_portion_start:]
        
        # Build character sequence
        char_sequence = []
        for features in latter_features:
            char = self.add_pattern_from_features(features)
            char_sequence.append(char)
        
        return "".join(char_sequence)
    
    def match_pattern_to_character(self, features: np.ndarray, threshold: float = 0.8) -> Optional[str]:
        """
        Match feature vector to existing character pattern.
        
        Args:
            features: Feature vector to match
            threshold: Similarity threshold (0-1)
            
        Returns:
            Matched character or None
        """
        if not self.pattern_templates:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for char, template in self.pattern_templates.items():
            # Compute cosine similarity
            similarity = np.dot(features, template) / (
                np.linalg.norm(features) * np.linalg.norm(template) + 1e-10
            )
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = char
        
        return best_match
    
    def get_character_set_info(self) -> Dict:
        """
        Get information about the built character set.
        
        Returns:
            Dictionary with character set information
        """
        return {
            "total_characters": len(self.character_set),
            "unique_patterns": len(self.mapping.pattern_to_char),
            "most_frequent": self.mapping.get_most_frequent_patterns(5),
            "character_set": "".join(self.character_set[:50])  # First 50 chars
        }
    
    def export_character_set(self) -> Dict[str, any]:
        """
        Export the complete character set and mappings.
        
        Returns:
            Dictionary containing all character set data
        """
        return {
            "characters": self.character_set,
            "pattern_mappings": self.mapping.pattern_to_char,
            "templates": {k: v.tolist() for k, v in self.pattern_templates.items()},
            "frequencies": self.mapping.pattern_frequency
        }
    
    def clear(self):
        """Clear all character set data."""
        self.mapping = PatternCharacterMapping()
        self.pattern_templates.clear()
        self.character_set.clear()
        self.next_char_index = 0
        print("Character set cleared")

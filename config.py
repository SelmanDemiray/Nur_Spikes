from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SNNConfig:
    """Configuration for SNN parameters."""
    dt: float = 0.005
    gain: float = 1.0
    nrep: int = 5
    batch_size: int = 32
    target_rate: float = 10
    adaptive_learning: bool = True
    use_homeostatic: bool = True
    log_level: str = "INFO"
    window_size: int = 100
    update_interval: int = 50
    prediction_threshold: float = 40.0
    threshold: float = 1.0  # Increased threshold to reduce excessive layer additions
    spike_threshold: float = 0.5  # Adjusted spike threshold for better spike management
    min_rate: float = 0.0  # Lowered minimum firing rate to handle lower spike rates
    max_rate: float = 100.0  # Maximum firing rate
    learning_rate: float = 0.001  # New field for learning rate
    enable_advanced_metrics: bool = True  # Enabled advanced metrics
    enable_stddev_metrics: bool = False  # New field for enabling standard deviation metrics
    max_dynamic_layers: int = 20  # Increased maximum dynamic layers allowed
    spike_threshold_layer_2: float = 0.5  # New field for Layer 2 spike threshold

    # Visualization parameters
    viz_update_interval: float = 0.05
    viz_window_size: int = 1000
    spike_scale: float = 5.0  # Scale factor for spike visualization
    rate_scale: float = 2.0   # Scale factor for rate visualization
    colormap_min: float = 0.0
    colormap_max: float = 1.0

    # Network parameters
    # input_size and output_size will be set dynamically based on weights

    @classmethod
    def from_weights(cls, W0: Any, W1: Any) -> 'SNNConfig':
        """Initialize config based on weight matrices."""
        input_size = W0.shape[1]
        hidden_size = W0.shape[0]
        output_size = W1.shape[0]
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        )
    
    def __post_init__(self):
        """Initialize parameters dictionary."""
        self.parameters = {
            "dt": self.dt,
            "gain": self.gain,
            "nrep": self.nrep,
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "spike_threshold": self.spike_threshold,
            "learning_rate": self.learning_rate,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "max_dynamic_layers": self.max_dynamic_layers,
            "spike_threshold_layer_2": self.spike_threshold_layer_2
        }

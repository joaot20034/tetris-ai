import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict

class TetrisFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for the Tetris Dict Observation Space.
    Extracts spatial features from the board using a CNN and combines 
    them with scalar metadata (current piece, next pieces, etc.).
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # 1. Board CNN
        # The board comes in as (20, 10). We add a channel dimension to make it (1, 20, 10)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: (32, 10, 5)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute the flattened size of the CNN output
        with torch.no_grad():
            dummy_board = torch.zeros(1, 1, 20, 10)
            n_flatten = self.cnn(dummy_board).shape[1]

        # 2. Scalar MLP (Metadata)
        # current_piece (1) + next_pieces (5) + hold_piece (1) + can_hold (1) = 8 scalar inputs
        n_scalars = 1 + 5 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process Board: cast to float, add channel dim (B, 1, H, W)
        board = observations["board"].float().unsqueeze(1)
        board_features = self.cnn(board)
        
        # Process Scalars: flatten next_pieces and concatenate everything
        current = observations["current_piece"].float().unsqueeze(1)
        next_p = observations["next_pieces"].float() # already shape (B, 5)
        hold = observations["hold_piece"].float().unsqueeze(1)
        can_hold = observations["can_hold"].float().unsqueeze(1)
        
        scalars = torch.cat([current, next_p, hold, can_hold], dim=1)
        scalar_features = self.mlp(scalars)
        
        # Concatenate CNN and MLP outputs
        combined = torch.cat([board_features, scalar_features], dim=1)
        return self.fusion(combined)
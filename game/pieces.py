import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# Standard Tetris Colors (R, G, B)
COLORS = {
    'I': (0, 255, 255),    # Cyan
    'J': (0, 0, 255),      # Blue
    'L': (255, 165, 0),    # Orange
    'O': (255, 255, 0),    # Yellow
    'S': (0, 255, 0),      # Green
    'T': (128, 0, 128),    # Purple
    'Z': (255, 0, 0),      # Red
    'GHOST': (50, 50, 50)  # Dark Gray
}

# 0 = Empty, 1 = Block
SHAPES = {
    'I': [[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
          [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
          [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]],
    
    'J': [[[1, 0, 0], [1, 1, 1], [0, 0, 0]],
          [[0, 1, 1], [0, 1, 0], [0, 1, 0]],
          [[0, 0, 0], [1, 1, 1], [0, 0, 1]],
          [[0, 1, 0], [0, 1, 0], [1, 1, 0]]],
    
    'L': [[[0, 0, 1], [1, 1, 1], [0, 0, 0]],
          [[0, 1, 0], [0, 1, 0], [0, 1, 1]],
          [[0, 0, 0], [1, 1, 1], [1, 0, 0]],
          [[1, 1, 0], [0, 1, 0], [0, 1, 0]]],
    
    'O': [[[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
          [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
          [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
          [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]],
    
    'S': [[[0, 1, 1], [1, 1, 0], [0, 0, 0]],
          [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
          [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
          [[1, 0, 0], [1, 1, 0], [0, 1, 0]]],
    
    'T': [[[0, 1, 0], [1, 1, 1], [0, 0, 0]],
          [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
          [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
          [[0, 1, 0], [1, 1, 0], [0, 1, 0]]],
    
    'Z': [[[1, 1, 0], [0, 1, 1], [0, 0, 0]],
          [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
          [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
          [[0, 1, 0], [1, 1, 0], [1, 0, 0]]]
}

@dataclass
class Tetromino:
    """Represents a single Tetris piece and its state."""
    name: str
    x: int = 3  # Spawns horizontally centered (10 wide grid -> col 3/4)
    y: int = 0  # Spawns at the top
    rotation: int = 0
    
    @property
    def shape(self) -> List[List[int]]:
        """Returns the current 2D grid of the piece based on its rotation."""
        return SHAPES[self.name][self.rotation]
    
    @property
    def color(self) -> Tuple[int, int, int]:
        return COLORS[self.name]
    
    def rotate(self, clockwise: bool = True) -> None:
        """Updates rotation state (0, 1, 2, 3)."""
        if clockwise:
            self.rotation = (self.rotation + 1) % 4
        else:
            self.rotation = (self.rotation - 1) % 4

class BagRandomizer:
    """Implements the modern Tetris 7-bag randomizer logic."""
    def __init__(self):
        self.queue: List[str] = []
        self._fill_bag()
        self._fill_bag()  # Keep 14 pieces so we can always see 'Next' queue
        
    def _fill_bag(self) -> None:
        """Shuffles a standard bag of 7 pieces and appends to queue."""
        bag = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
        random.shuffle(bag)
        self.queue.extend(bag)
        
    def pop(self) -> str:
        """Returns the next piece name and maintains the queue."""
        piece = self.queue.pop(0)
        if len(self.queue) <= 7:
            self._fill_bag()
        return piece
    
    def get_queue(self, n: int = 5) -> List[str]:
        """Returns the next 'n' pieces for the UI preview."""
        return self.queue[:n]
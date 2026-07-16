import pygame
from dataclasses import dataclass

@dataclass
class TrainingState:
    paused: bool = False
    speed_multiplier: int = 1  # 1x, 2x, 5x, 10x, 100x, 1000x
    fps_base: int = 60

class KeyboardControls:
    """Handles user input to manipulate training speed and state."""
    
    SPEEDS = [1, 2, 5, 10, 100, 1000]
    
    def __init__(self):
        self.state = TrainingState()
        self.current_speed_idx = 0

    def process_events(self) -> bool:
        """
        Reads Pygame events. 
        Returns False if the user closed the window (triggering a training abort), True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.state.paused = not self.state.paused
                    
                elif event.key == pygame.K_UP:
                    self.current_speed_idx = min(len(self.SPEEDS) - 1, self.current_speed_idx + 1)
                    self.state.speed_multiplier = self.SPEEDS[self.current_speed_idx]
                    
                elif event.key == pygame.K_DOWN:
                    self.current_speed_idx = max(0, self.current_speed_idx - 1)
                    self.state.speed_multiplier = self.SPEEDS[self.current_speed_idx]
                    
        return True
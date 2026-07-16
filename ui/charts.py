import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, crucial for Pygame integration
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame
import numpy as np
from typing import List

class RealTimePlotter:
    """Renders Matplotlib charts to Pygame Surfaces."""
    
    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height
        
        # Create Figure and Canvas
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.surface: pygame.Surface = None
        
        self._format_axis()

    def _format_axis(self):
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

    def update_plot(self, data: List[float], title: str = "Reward over Time", color: str = '#00ff00') -> pygame.Surface:
        """Draws the plot and converts it to a Pygame surface."""
        self._format_axis()
        
        if len(data) > 0:
            # Smooth the curve for better readability
            if len(data) > 10:
                smoothed = np.convolve(data, np.ones(10)/10, mode='valid')
                self.ax.plot(range(len(smoothed)), smoothed, color=color, linewidth=2)
            else:
                self.ax.plot(data, color=color, linewidth=2)
                
        self.ax.set_title(title, color='white')
        self.fig.tight_layout()
        
        # Render to canvas and convert to Pygame Surface
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()
        
        self.surface = pygame.image.fromstring(raw_data, size, "RGB")
        return self.surface
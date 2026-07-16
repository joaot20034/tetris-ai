import pygame
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from ui.controls import KeyboardControls
from ui.charts import RealTimePlotter

class UIDashboardCallback(BaseCallback):
    """
    SB3 Callback that renders the Tetris environment, handles user controls, 
    and overlays training telemetry and Matplotlib charts.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.controls = KeyboardControls()
        self.plotter = RealTimePlotter(width=400, height=250)
        self.clock = pygame.time.Clock()
        
        # Telemetry
        self.episode_rewards = []
        self.current_reward = 0.0
        self.episodes = 0
        self.best_reward = -float('inf')
        
        # Cached UI Elements
        self.font = None
        self.chart_surface = None

    def _on_training_start(self) -> None:
        """Initialize Pygame extensions once training begins."""
        pygame.init()
        self.font = pygame.font.SysFont('consolas', 18)
        self.chart_surface = self.plotter.update_plot([])

    def _on_step(self) -> bool:
        """Called by SB3 at every single environment step."""
        
        # 1. Process Controls (Pause / Speed / Quit)
        if not self.controls.process_events():
            return False  # Cancels training
            
        while self.controls.state.paused:
            if not self.controls.process_events():
                return False
            self.clock.tick(10)  # Keep loop alive but idle

        # 2. Extract Data from VecEnv
        # SB3 wraps envs in VecEnv. We get the raw TetrisEnv.
        env = self.training_env.envs[0].unwrapped
        
        # Accumulate reward
        self.current_reward += self.locals["rewards"][0]
        
        # Check if episode ended
        if self.locals["dones"][0]:
            self.episodes += 1
            self.episode_rewards.append(self.current_reward)
            if self.current_reward > self.best_reward:
                self.best_reward = self.current_reward
            self.current_reward = 0.0
            
            # Update chart every episode to prevent lag
            self.chart_surface = self.plotter.update_plot(self.episode_rewards)

        # 3. Handle Rendering & Speed
        speed = self.controls.state.speed_multiplier
        step_count = self.num_timesteps
        
        # Only render every Nth frame based on speed multiplier to save CPU
        if step_count % speed == 0:
            env.render()
            
            # Apply FPS cap if we are running at normal visual speeds (1x, 2x, 5x)
            if speed <= 5:
                self.clock.tick(self.controls.state.fps_base * speed)
                
            # If the environment successfully rendered a Pygame window, overlay our UI
            if env.renderer and env.renderer.window:
                self._draw_overlay(env.renderer.window, env)
                pygame.display.flip()
                
        return True

    def _draw_overlay(self, surface: pygame.Surface, env) -> None:
        """Draws the Dashboard on top of the game window."""
        
        # Define layout coordinates
        ui_x = surface.get_width() - 420  # Draw on the far right
        ui_y = 20
        
        # Text Metrics
        metrics = [
            f"EPISODE: {self.episodes}",
            f"TIMESTEPS: {self.num_timesteps}",
            f"BEST REW: {self.best_reward:.1f}",
            f"SPEED: {self.controls.state.speed_multiplier}x " + ("(PAUSED)" if self.controls.state.paused else ""),
            "CONTROLS: [UP/DOWN] Speed, [SPACE] Pause"
        ]
        
        # Draw semi-transparent background for text
        bg_rect = pygame.Rect(ui_x - 10, ui_y - 10, 400, len(metrics) * 25 + 20)
        s = pygame.Surface((bg_rect.width, bg_rect.height))
        s.set_alpha(200)
        s.fill((0, 0, 0))
        surface.blit(s, bg_rect.topleft)
        
        for i, text in enumerate(metrics):
            text_surf = self.font.render(text, True, (255, 255, 255))
            surface.blit(text_surf, (ui_x, ui_y + (i * 25)))
            
        # Draw Chart
        if self.chart_surface:
            chart_y = bg_rect.bottom + 20
            surface.blit(self.chart_surface, (ui_x - 10, chart_y))
from typing import Dict

class ScoreCalculator:
    """
    Calculates standard Guideline Tetris scores.
    Separated from the core game loop to keep the physics engine clean.
    """
    
    # Standard Guideline scoring multipliers
    LINE_MULTIPLIERS: Dict[int, int] = {
        1: 100,   # Single
        2: 300,   # Double
        3: 500,   # Triple
        4: 800    # Tetris
    }

    @staticmethod
    def calculate_line_score(lines_cleared: int, level: int) -> int:
        """Calculates the score awarded for clearing lines."""
        if lines_cleared == 0:
            return 0
            
        # Standard Tetris caps at 4 lines (a "Tetris")
        lines = min(lines_cleared, 4)
        return ScoreCalculator.LINE_MULTIPLIERS.get(lines, 0) * level

    @staticmethod
    def calculate_drop_score(cells_dropped: int, is_hard_drop: bool = False) -> int:
        """Calculates the score for soft and hard drops."""
        if is_hard_drop:
            return cells_dropped * 2
        return cells_dropped
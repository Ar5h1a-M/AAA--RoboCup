import numpy as np

def GenerateBasicFormation():
    """
    Generates an aggressive formation:
    - Player 1 (Goalkeeper): Stays near own goal
    - Player 2 (Defender): Defensive position
    - Players 3-5 (Attackers): Spread across attacking positions
    """
    formation = np.array([
        [-14.0, 0.0],    # Player 1: Goalkeeper at own goal
        [-8.0, 0.0],     # Player 2: Defender in defensive third
        [2.0, -3.0],     # Player 3: Attacker left
        [2.0, 3.0],      # Player 4: Attacker right  
        [5.0, 0.0]       # Player 5: Attacker center/forward
    ])
    
    return formation
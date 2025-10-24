import numpy as np

class DynamicFormation:
    """Generates dynamic formations based on ball position and game state"""
    
    @staticmethod
    def generate_formation(ball_pos, play_mode_group, team_side_is_left=True):
        """
        Generate formation based on ball position and game phase
        
        Parameters:
        -----------
        ball_pos : ndarray
            Current ball position [x, y]
        play_mode_group : int
            Current play mode group
        team_side_is_left : bool
            True if team plays on left side
            
        Returns:
        --------
        formation : list of ndarray
            List of 5 positions for players 1-5
        """
        ball_x, ball_y = ball_pos[0], ball_pos[1]
        
        # Determine formation type based on ball position
        if ball_x > 5:  # Ball in opponent's half - attacking formation
            return DynamicFormation._attacking_formation(ball_x, ball_y)
        elif ball_x > -5:  # Ball in midfield - balanced formation
            return DynamicFormation._balanced_formation(ball_x, ball_y)
        else:  # Ball in our half - defensive formation
            return DynamicFormation._defensive_formation(ball_x, ball_y)
    
    @staticmethod
    def _attacking_formation(ball_x, ball_y):
        """Aggressive formation when ball is in opponent's half"""
        formation = [
            np.array([-10, 0]),           # Goalkeeper - slightly forward
            np.array([-2, -3]),           # Defender Left
            np.array([ball_x - 3, ball_y]),  # Support - follows ball
            np.array([ball_x + 2, ball_y + 3 if ball_y < 0 else ball_y - 3]),  # Wing forward
            np.array([13, 0])             # Striker
        ]
        return formation
    
    @staticmethod
    def _balanced_formation(ball_x, ball_y):
        """Balanced formation for midfield play"""
        formation = [
            np.array([-13, 0]),           # Goalkeeper
            np.array([-7, -3]),           # Defender Left
            np.array([-7, 3]),            # Defender Right
            np.array([ball_x, ball_y + 3 if ball_y < 0 else ball_y - 3]),  # Midfielder
            np.array([8, 0])              # Forward
        ]
        return formation
    
    @staticmethod
    def _defensive_formation(ball_x, ball_y):
        """Defensive formation when ball is in our half"""
        # All players shift back to defend
        formation = [
            np.array([-13, 0]),           # Goalkeeper
            np.array([-9, -4]),           # Defender Left
            np.array([-9, 4]),            # Defender Right
            np.array([ball_x + 2, ball_y]),  # Midfielder - slightly ahead of ball
            np.array([-3, 0])             # Forward - drops back
        ]
        return formation
    
    @staticmethod
    def generate_goalkeeper_position(ball_pos, goal_pos=np.array([-15, 0])):
        """
        Calculate optimal goalkeeper position
        
        Parameters:
        -----------
        ball_pos : ndarray
            Ball position [x, y]
        goal_pos : ndarray
            Goal center position [x, y]
            
        Returns:
        --------
        gk_pos : ndarray
            Goalkeeper position [x, y]
        """
        # Goalkeeper stays on line between ball and goal center
        ball_to_goal = goal_pos - ball_pos
        distance = np.linalg.norm(ball_to_goal)
        
        if distance < 0.1:
            return np.array([-13, 0])
        
        direction = ball_to_goal / distance
        
        # Position goalkeeper 2m from goal line
        gk_pos = goal_pos + direction * 2.0
        
        # Clamp to reasonable area
        gk_pos[0] = np.clip(gk_pos[0], -14, -11)
        gk_pos[1] = np.clip(gk_pos[1], -3, 3)
        
        return gk_pos
    
    @staticmethod
    def adjust_formation_for_opponents(formation, opponent_positions, ball_pos):
        """
        Adjust formation to account for opponent positions (man-marking)
        
        Parameters:
        -----------
        formation : list of ndarray
            Base formation positions
        opponent_positions : list
            List of opponent positions
        ball_pos : ndarray
            Current ball position
            
        Returns:
        --------
        adjusted_formation : list of ndarray
            Formation adjusted for opponents
        """
        adjusted = []
        
        for i, pos in enumerate(formation):
            if i == 0:  # Don't adjust goalkeeper much
                adjusted.append(pos)
                continue
            
            # Find closest opponent
            closest_opp = None
            min_dist = float('inf')
            
            for opp_pos in opponent_positions:
                if opp_pos is None:
                    continue
                dist = np.linalg.norm(pos - opp_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_opp = opp_pos
            
            # If opponent is very close, adjust position to mark them
            if closest_opp is not None and min_dist < 3.0:
                # Move slightly toward opponent for marking
                to_opponent = closest_opp - pos
                adjusted_pos = pos + to_opponent * 0.3
                adjusted.append(adjusted_pos)
            else:
                adjusted.append(pos)
        
        return adjusted
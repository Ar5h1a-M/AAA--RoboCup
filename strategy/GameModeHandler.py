import numpy as np
from math_ops.Math_Ops import Math_Ops as M

class GameModeHandler:
    """Handles positioning and behaviors for different game modes"""
    
    def __init__(self, world):
        self.world = world
        
    def get_set_piece_behavior(self, strategyData):
        """
        Determine behavior during set pieces (kickoffs, free kicks, etc.)
        
        Returns:
        --------
        action_type : str
            'move' or 'kick'
        target : tuple
            target position or kick target
        orientation : float or None
            desired orientation
        """
        W = self.world
        PM = strategyData.play_mode
        my_unum = strategyData.player_unum
        
        # Our kickoff
        if PM == W.M_OUR_KICKOFF:
            if my_unum == 1:  # Goalkeeper stays back
                return 'move', np.array([-13, 0]), 0
            elif my_unum == strategyData.active_player_unum:
                # Active player kicks off
                if strategyData.ball_dist < 0.5:
                    return 'kick', np.array([0, 3]), None  # Pass to side
                else:
                    return 'move', strategyData.ball_2d, strategyData.ball_dir
            else:
                # Others wait in formation slightly behind ball
                return 'move', strategyData.my_desired_position, strategyData.ball_dir
        
        # Their kickoff - defensive positioning
        elif PM == W.M_THEIR_KICKOFF:
            if my_unum == 1:  # Goalkeeper
                return 'move', np.array([-13, 0]), 0
            else:
                # Form defensive line
                defensive_pos = strategyData.my_desired_position.copy()
                defensive_pos[0] = min(defensive_pos[0], -1)  # Stay behind center line
                return 'move', defensive_pos, strategyData.ball_dir
        
        # Our free kick / corner / goal kick
        elif PM in [W.M_OUR_FREE_KICK, W.M_OUR_CORNER_KICK, W.M_OUR_GOAL_KICK]:
            if my_unum == strategyData.active_player_unum:
                if strategyData.ball_dist < 0.5:
                    # Find best teammate to pass to
                    target = self._find_best_pass_target(strategyData)
                    return 'kick', target, None
                else:
                    return 'move', strategyData.ball_2d, strategyData.ball_dir
            else:
                # Position for receiving
                return 'move', strategyData.my_desired_position, strategyData.ball_dir
        
        # Their free kick / corner / goal kick - defensive
        elif PM in [W.M_THEIR_FREE_KICK, W.M_THEIR_CORNER_KICK, W.M_THEIR_GOAL_KICK]:
            if my_unum == 1:  # Goalkeeper
                return 'move', np.array([-13, 0]), strategyData.ball_dir
            else:
                # Mark opponents or cover space
                defensive_pos = self._get_defensive_position(strategyData)
                return 'move', defensive_pos, strategyData.ball_dir
        
        # Default: return to formation
        return 'move', strategyData.my_desired_position, strategyData.ball_dir
    
    def _find_best_pass_target(self, strategyData):
        """Find the best teammate to pass to"""
        best_target = np.array([15, 0])  # Default: goal
        best_score = -1000
        
        for i, pos in enumerate(strategyData.teammate_positions):
            if pos is None or i == strategyData.player_unum - 1:
                continue
            
            # Score based on: distance to goal, openness, distance from passer
            dist_to_goal = np.linalg.norm(pos - np.array([15, 0]))
            dist_from_me = np.linalg.norm(pos - strategyData.mypos)
            
            # Check if path is clear of opponents
            path_clear = self._is_path_clear(strategyData.mypos, pos, strategyData.opponent_positions)
            
            score = -dist_to_goal + (10 if path_clear else 0) - dist_from_me * 0.5
            
            if score > best_score:
                best_score = score
                best_target = pos
        
        return best_target
    
    def _get_defensive_position(self, strategyData):
        """Calculate defensive position to mark opponents or cover space"""
        ball_pos = strategyData.ball_2d
        my_goal = np.array([-15, 0])
        
        # Position between ball and goal
        direction = my_goal - ball_pos
        direction = direction / (np.linalg.norm(direction) + 0.001)
        
        # Stay closer to goal than ball
        defensive_pos = ball_pos + direction * 2.0
        defensive_pos[0] = max(defensive_pos[0], -14)  # Don't go behind goal line
        
        return defensive_pos
    
    def _is_path_clear(self, start, end, obstacles):
        """Check if path between two points is clear of obstacles"""
        if not obstacles:
            return True
        
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        
        if path_len < 0.01:
            return True
        
        path_dir = path_vec / path_len
        
        for obs in obstacles:
            if obs is None:
                continue
            
            # Vector from start to obstacle
            to_obs = obs - start
            
            # Project onto path
            projection = np.dot(to_obs, path_dir)
            
            # Check if obstacle is along the path
            if 0 < projection < path_len:
                # Distance from obstacle to path
                closest_point = start + path_dir * projection
                dist = np.linalg.norm(obs - closest_point)
                
                if dist < 1.0:  # Obstacle blocks path
                    return False
        
        return True
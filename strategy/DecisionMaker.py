import numpy as np
from math_ops.Math_Ops import Math_Ops as M

class DecisionMaker:
    """Makes tactical decisions for active player with ball"""
    
    def __init__(self, world):
        self.world = world
    
    def decide_action(self, strategyData):
        """
        Decide whether to shoot, pass, or dribble
        
        Returns:
        --------
        action : str
            'shoot', 'pass', or 'dribble'
        target : ndarray
            Target position for the action
        """
        
        # Analyze situation
        shoot_quality = self._evaluate_shot(strategyData)
        pass_quality, best_pass_target = self._evaluate_pass(strategyData)
        dribble_quality = self._evaluate_dribble(strategyData)
        
        # Decision making based on quality scores
        if shoot_quality > 0.7 and shoot_quality > pass_quality:
            return 'shoot', np.array([15, 0])
        elif pass_quality > 0.5 and pass_quality > dribble_quality:
            return 'pass', best_pass_target
        else:
            return 'dribble', self._get_dribble_target(strategyData)
    
    def _evaluate_shot(self, strategyData):
        """
        Evaluate quality of shooting opportunity
        
        Returns:
        --------
        quality : float [0, 1]
            Quality score for shooting
        """
        goal_pos = np.array([15, 0])
        my_pos = strategyData.mypos
        ball_pos = strategyData.ball_2d
        
        # Distance to goal (closer is better)
        dist_to_goal = np.linalg.norm(ball_pos - goal_pos)
        distance_score = max(0, 1 - dist_to_goal / 20.0)
        
        # Angle to goal (more centered is better)
        angle_to_goal = abs(strategyData.goal_dir)
        angle_score = max(0, 1 - angle_to_goal / 90.0)
        
        # Check for opponents blocking shot
        opponents_blocking = self._count_opponents_in_path(
            ball_pos, goal_pos, strategyData.opponent_positions, radius=1.5
        )
        blocking_score = max(0, 1 - opponents_blocking * 0.4)
        
        # Only shoot if reasonably close to goal
        if dist_to_goal > 12:
            return 0.0
        
        # Combined score
        quality = (distance_score * 0.4 + angle_score * 0.3 + blocking_score * 0.3)
        
        return quality
    
    def _evaluate_pass(self, strategyData):
        """
        Evaluate passing opportunities
        
        Returns:
        --------
        quality : float [0, 1]
            Quality score for best pass
        target : ndarray
            Position of best pass target
        """
        best_quality = 0.0
        best_target = np.array([15, 0])
        
        for i, teammate_pos in enumerate(strategyData.teammate_positions):
            if teammate_pos is None or i == strategyData.player_unum - 1:
                continue
            
            quality = self._evaluate_pass_to_teammate(strategyData, teammate_pos, i)
            
            if quality > best_quality:
                best_quality = quality
                best_target = teammate_pos
        
        return best_quality, best_target
    
    def _evaluate_pass_to_teammate(self, strategyData, teammate_pos, teammate_idx):
        """Evaluate quality of pass to specific teammate"""
        my_pos = strategyData.mypos
        goal_pos = np.array([15, 0])
        
        # Is teammate ahead of me? (progressive pass)
        forward_progress = teammate_pos[0] - my_pos[0]
        progress_score = np.clip(forward_progress / 10.0, 0, 1)
        
        # Distance to goal from teammate
        teammate_dist_to_goal = np.linalg.norm(teammate_pos - goal_pos)
        goal_proximity_score = max(0, 1 - teammate_dist_to_goal / 25.0)
        
        # Is pass path clear?
        opponents_in_path = self._count_opponents_in_path(
            my_pos, teammate_pos, strategyData.opponent_positions, radius=1.0
        )
        clear_path_score = max(0, 1 - opponents_in_path * 0.5)
        
        # Distance of pass (not too far)
        pass_distance = np.linalg.norm(teammate_pos - my_pos)
        distance_score = max(0, 1 - pass_distance / 15.0)
        
        # Is teammate under pressure?
        pressure_score = 1.0
        for opp_pos in strategyData.opponent_positions:
            if opp_pos is not None:
                dist_to_opp = np.linalg.norm(teammate_pos - opp_pos)
                if dist_to_opp < 2.0:
                    pressure_score *= 0.5
        
        # Combined score
        quality = (progress_score * 0.25 + 
                  goal_proximity_score * 0.2 + 
                  clear_path_score * 0.35 + 
                  distance_score * 0.1 +
                  pressure_score * 0.1)
        
        return quality
    
    def _evaluate_dribble(self, strategyData):
        """
        Evaluate dribbling opportunity
        
        Returns:
        --------
        quality : float [0, 1]
            Quality score for dribbling
        """
        my_pos = strategyData.mypos
        
        # Check space ahead
        space_ahead = self._check_space_ahead(strategyData)
        
        # Pressure from opponents
        pressure = 0
        for opp_pos in strategyData.opponent_positions:
            if opp_pos is not None:
                dist = np.linalg.norm(my_pos - opp_pos)
                if dist < 3.0:
                    pressure += (3.0 - dist) / 3.0
        
        pressure_score = max(0, 1 - pressure * 0.5)
        
        # Combined score
        quality = (space_ahead * 0.6 + pressure_score * 0.4)
        
        return quality
    
    def _check_space_ahead(self, strategyData):
        """Check if there's space to dribble forward"""
        my_pos = strategyData.mypos
        my_ori = strategyData.my_ori
        
        # Check points ahead
        forward_dir = np.array([np.cos(np.radians(my_ori)), 
                               np.sin(np.radians(my_ori))])
        
        check_distance = 3.0
        check_pos = my_pos + forward_dir * check_distance
        
        # Count opponents near target position
        opponents_near = 0
        for opp_pos in strategyData.opponent_positions:
            if opp_pos is not None:
                dist = np.linalg.norm(check_pos - opp_pos)
                if dist < 2.0:
                    opponents_near += 1
        
        return max(0, 1 - opponents_near * 0.5)
    
    def _get_dribble_target(self, strategyData):
        """Get target position for dribbling"""
        my_pos = strategyData.mypos
        goal_pos = np.array([15, 0])
        
        # Dribble toward goal but avoid opponents
        to_goal = goal_pos - my_pos
        distance = np.linalg.norm(to_goal)
        
        if distance < 0.1:
            return goal_pos
        
        direction = to_goal / distance
        
        # Target 2-3 meters ahead
        target = my_pos + direction * 2.5
        
        # Adjust if opponents are in the way
        target = self._avoid_opponents_on_path(my_pos, target, strategyData.opponent_positions)
        
        return target
    
    def _count_opponents_in_path(self, start, end, opponent_positions, radius=1.0):
        """Count opponents within radius of path between start and end"""
        count = 0
        path_vec = end - start
        path_len = np.linalg.norm(path_vec)
        
        if path_len < 0.01:
            return 0
        
        path_dir = path_vec / path_len
        
        for opp_pos in opponent_positions:
            if opp_pos is None:
                continue
            
            to_opp = opp_pos - start
            projection = np.dot(to_opp, path_dir)
            
            if 0 < projection < path_len:
                closest_point = start + path_dir * projection
                dist = np.linalg.norm(opp_pos - closest_point)
                
                if dist < radius:
                    count += 1
        
        return count
    
    def _avoid_opponents_on_path(self, start, target, opponent_positions):
        """Adjust target to avoid opponents"""
        adjusted_target = target.copy()
        
        for opp_pos in opponent_positions:
            if opp_pos is None:
                continue
            
            # Check if opponent is near target
            dist_to_target = np.linalg.norm(opp_pos - target)
            
            if dist_to_target < 2.0:
                # Shift target perpendicular to avoid opponent
                to_target = target - start
                perpendicular = np.array([-to_target[1], to_target[0]])
                perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 0.001)
                
                adjusted_target += perpendicular * 1.5
        
        # Clamp to field boundaries
        adjusted_target[0] = np.clip(adjusted_target[0], -14, 14)
        adjusted_target[1] = np.clip(adjusted_target[1], -9, 9)
        
        return adjusted_target
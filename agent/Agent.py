from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation
from strategy.DynamicFormation import DynamicFormation
from strategy.GameModeHandler import GameModeHandler
from strategy.DecisionMaker import DecisionMaker


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



        



    def select_skill(self, strategyData):
        """
        Main decision-making function for robot behavior
        Handles all game modes and situations
        """
        drawer = self.world.draw
        W = self.world
        
        # ==============================================================
        # 1. HANDLE SET PIECES (Non-Play On modes)
        # ==============================================================
        if strategyData.PM_GROUP != W.MG_OTHER:  # Not play on or game over
            action_type, target, orientation = self.game_mode_handler.get_set_piece_behavior(strategyData)
            
            if action_type == 'kick':
                drawer.annotation((0, 10.5), "Set Piece: Kicking", drawer.Color.yellow, "status")
                drawer.line(strategyData.mypos, target, 2, drawer.Color.red, "kick_line")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:  # action_type == 'move'
                drawer.annotation((0, 10.5), "Set Piece: Positioning", drawer.Color.yellow, "status")
                drawer.line(strategyData.mypos, target, 2, drawer.Color.blue, "move_line")
                return self.move(target, orientation=orientation)
        
        # ==============================================================
        # 2. PLAY ON MODE - DYNAMIC FORMATION & ROLE ASSIGNMENT
        # ==============================================================
        
        # Generate dynamic formation based on ball position
        formation_positions = DynamicFormation.generate_formation(
            strategyData.ball_2d, 
            strategyData.PM_GROUP,
            self.world.team_side_is_left
        )
        
        # Adjust formation for opponent positions (man-marking)
        formation_positions = DynamicFormation.adjust_formation_for_opponents(
            formation_positions,
            strategyData.opponent_positions,
            strategyData.ball_2d
        )
        
        # Assign roles using Hungarian algorithm
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )
        
        # Draw formation assignment
        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target_line")
        
        # ==============================================================
        # 3. GOALKEEPER SPECIAL BEHAVIOR (Player 1)
        # ==============================================================
        if strategyData.player_unum == 1:
            return self._goalkeeper_behavior(strategyData, drawer)
        
        # ==============================================================
        # 4. ACTIVE PLAYER BEHAVIOR (Closest to ball)
        # ==============================================================
        if strategyData.active_player_unum == strategyData.player_unum:
            drawer.annotation((0, 10.5), "Active Player: Attacking", drawer.Color.yellow, "status")
            return self._active_player_behavior(strategyData, drawer)
        
        # ==============================================================
        # 5. SUPPORTING PLAYER BEHAVIOR (Not active)
        # ==============================================================
        else:
            drawer.clear("status")
            return self._supporting_player_behavior(strategyData, drawer)
    
    
    def _goalkeeper_behavior(self, strategyData, drawer):
        """Goalkeeper-specific behavior"""
        ball_pos = strategyData.ball_2d
        my_goal = np.array([-15, 0])
        
        # Calculate optimal goalkeeper position
        gk_position = DynamicFormation.generate_goalkeeper_position(ball_pos, my_goal)
        
        # If ball is very close and we're the closest, go for it
        if strategyData.ball_dist < 3.0 and strategyData.active_player_unum == 1:
            drawer.annotation((0, 10.5), "Goalkeeper: Clearing Ball", drawer.Color.red, "status")
            
            # Decide where to clear the ball
            if ball_pos[0] < -10:  # Ball very close to goal
                # Clear to sides
                clear_target = np.array([0, 8 if ball_pos[1] > 0 else -8])
            else:
                # Boot it forward
                clear_target = np.array([10, ball_pos[1]])
            
            drawer.line(strategyData.mypos, clear_target, 2, drawer.Color.red, "kick_line")
            return self.kickTarget(strategyData, strategyData.mypos, clear_target)
        else:
            # Stay in position
            drawer.annotation((0, 10.5), "Goalkeeper: Positioning", drawer.Color.cyan, "status")
            drawer.line(strategyData.mypos, gk_position, 2, drawer.Color.cyan, "gk_line")
            return self.move(gk_position, orientation=strategyData.ball_dir)
    
    
    def _active_player_behavior(self, strategyData, drawer):
        """Behavior for player closest to ball (active player)"""
        
        # First, check if we need to reach the ball
        if strategyData.ball_dist > 0.5:
            drawer.annotation((0, 10.5), "Active: Approaching Ball", drawer.Color.orange, "status")
            drawer.line(strategyData.mypos, strategyData.ball_2d, 2, drawer.Color.orange, "approach_line")
            return self.move(strategyData.ball_2d, orientation=strategyData.ball_dir)
        
        # We're close to the ball - make tactical decision
        action, target = self.decision_maker.decide_action(strategyData)
        
        if action == 'shoot':
            drawer.annotation((0, 10.5), "Active: SHOOTING!", drawer.Color.red, "status")
            drawer.line(strategyData.mypos, target, 3, drawer.Color.red, "shot_line")
            return self.kickTarget(strategyData, strategyData.mypos, target, enable_pass_command=True)
        
        elif action == 'pass':
            drawer.annotation((0, 10.5), "Active: Passing", drawer.Color.green, "status")
            drawer.line(strategyData.mypos, target, 2, drawer.Color.green, "pass_line")
            drawer.circle(target, 0.5, drawer.Color.green, "pass_target")
            return self.kickTarget(strategyData, strategyData.mypos, target, enable_pass_command=True)
        
        else:  # action == 'dribble'
            drawer.annotation((0, 10.5), "Active: Dribbling", drawer.Color.yellow, "status")
            drawer.line(strategyData.mypos, target, 2, drawer.Color.yellow, "dribble_line")
            
            # Check if opponents are very close - if so, kick away quickly
            urgent_threat = False
            for opp_pos in strategyData.opponent_positions:
                if opp_pos is not None:
                    if np.linalg.norm(strategyData.mypos - opp_pos) < 1.5:
                        urgent_threat = True
                        break
            
            if urgent_threat:
                # Quick clearance
                drawer.annotation((0, 10.5), "Active: Under Pressure!", drawer.Color.red, "status")
                return self.kickTarget(strategyData, strategyData.mypos, target, enable_pass_command=False)
            else:
                # Normal dribble
                return self.move(target, orientation=None, is_aggressive=True)
    
    
    def _supporting_player_behavior(self, strategyData, drawer):
        """Behavior for supporting players (not active)"""
        
        # Check if we're in formation position
        distance_to_position = np.linalg.norm(
            strategyData.mypos - strategyData.my_desired_position
        )
        
        # Determine if we should mark an opponent or move to formation
        mark_target = self._find_opponent_to_mark(strategyData)
        
        if mark_target is not None:
            # Mark opponent
            drawer.annotation((0, 10.5), "Supporting: Marking", drawer.Color.magenta, "status")
            drawer.line(strategyData.mypos, mark_target, 2, drawer.Color.magenta, "mark_line")
            drawer.circle(mark_target, 0.8, drawer.Color.magenta, "mark_circle")
            return self.move(mark_target, orientation=strategyData.ball_dir)
        
        elif distance_to_position > 0.5:
            # Move to formation position
            drawer.annotation((0, 10.5), "Supporting: Moving to Position", drawer.Color.blue, "status")
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        
        else:
            # In position - face ball and be ready
            drawer.annotation((0, 10.5), "Supporting: In Position", drawer.Color.cyan, "status")
            drawer.clear("target_line")
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
    
    
    def _find_opponent_to_mark(self, strategyData):
        """
        Find an opponent to mark if they're dangerous
        
        Returns:
        --------
        mark_position : ndarray or None
            Position to move to for marking, or None if no marking needed
        """
        my_pos = strategyData.mypos
        ball_pos = strategyData.ball_2d
        my_goal = np.array([-15, 0])
        
        # Only mark if ball is in our half or midfield
        if ball_pos[0] > 5:
            return None
        
































    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")
from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1]


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:]
        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1])))
        else:
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        r = self.world.robot

        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute)
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)


    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        return self.behavior.execute("Dribble",None,None)


    def kickTarget(self, strategyData, mypos_2d=(0,0), target_2d=(0,0), abort=False, enable_pass_command=False):
        # Calculate kick direction towards target
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        kick_distance = np.linalg.norm(vector_to_target)
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        kick_direction = np.degrees(direction_radians)

        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = kick_direction
        self.kick_distance = kick_distance

        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
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
            self.beam(True)
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass

        self.radio.broadcast()

        if self.fat_proxy_cmd is None:
            self.scom.commit_and_send(strategyData.robot_model.get_command())
        else:
            self.scom.commit_and_send(self.fat_proxy_cmd.encode())
            self.fat_proxy_cmd = ""


    def select_skill(self, strategyData):
        drawer = self.world.draw
        
        # Define roles based on player number
        GOALKEEPER = 1
        DEFENDER = 2
        ATTACKERS = [3, 4, 5]
        
        my_unum = strategyData.player_unum
        opponent_goal = np.array([15.0, 0.0])
        is_active = (strategyData.active_player_unum == my_unum)
        
        # Role Assignment for positioning
        formation_positions = GenerateBasicFormation()
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[my_unum]
        
        # Draw formation position
        drawer.circle(strategyData.my_desired_position, 0.3, 2, False, drawer.Color.cyan, f"formation_{my_unum}")
        
        # ========== GOALKEEPER ==========
        if my_unum == GOALKEEPER:
            own_goal = np.array([-15.0, 0.0])
            ball_pos = strategyData.ball_2d
            
            # Position between ball and own goal
            direction_to_ball = ball_pos - own_goal
            dist_to_ball = np.linalg.norm(direction_to_ball)
            
            if dist_to_ball > 0.1:
                # Stay 2-3 meters in front of goal
                desired_pos = own_goal + (direction_to_ball / dist_to_ball) * min(3.0, dist_to_ball * 0.4)
                # Clamp Y position to stay in goal area
                desired_pos[1] = np.clip(desired_pos[1], -2.0, 2.0)
                desired_pos[0] = max(desired_pos[0], -14.5)  # Don't go past goal line
                strategyData.my_desired_position = desired_pos
            
            # Only attack ball if very close
            if strategyData.ball_dist < 0.6 and is_active:
                drawer.annotation((0, 10.5), "GK - CLEARING BALL!", drawer.Color.red, "status")
                drawer.line(strategyData.mypos, opponent_goal, 3, drawer.Color.red, "kick_line")
                return self.kickTarget(strategyData, strategyData.mypos, opponent_goal)
            else:
                drawer.annotation((0, 10.5), "GK - Defending", drawer.Color.blue, "status")
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        
        # ========== DEFENDER ==========
        elif my_unum == DEFENDER:
            ball_pos = strategyData.ball_2d
            
            # Only chase ball if I'm the active player
            if is_active:
                drawer.annotation((0, 10.5), "DEFENDER - ATTACKING!", drawer.Color.red, "status")
                drawer.line(strategyData.mypos, opponent_goal, 3, drawer.Color.red, "kick_line")
                return self.kickTarget(strategyData, strategyData.mypos, opponent_goal)
            else:
                # Stay in defensive position, don't chase
                drawer.annotation((0, 10.5), "DEFENDER - Holding Position", drawer.Color.yellow, "status")
                
                # Adjust position based on ball location
                ball_x = ball_pos[0]
                if ball_x < -5:  # Ball in defensive half
                    defensive_pos = np.array([-10.0, ball_pos[1] * 0.5])  # Track ball Y position slightly
                else:  # Ball in attacking half
                    defensive_pos = np.array([-6.0, 0.0])  # Hold center
                
                strategyData.my_desired_position = defensive_pos
                drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.yellow, f"move_{my_unum}")
                
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        
        # ========== ATTACKERS ==========
        elif my_unum in ATTACKERS:
            ball_pos = strategyData.ball_2d
            
            # Only the active player attacks the ball
            if is_active:
                drawer.annotation((0, 10.5), f"ATTACKER {my_unum} - SHOOTING!", drawer.Color.red, "status")
                drawer.line(strategyData.mypos, opponent_goal, 3, drawer.Color.red, "kick_line")
                
                # Kick towards goal
                return self.kickTarget(strategyData, strategyData.mypos, opponent_goal)
            
            else:
                # Hold attacking formation position
                drawer.annotation((0, 10.5), f"ATTACKER {my_unum} - Supporting", drawer.Color.green, "status")
                
                # Adjust formation position based on ball
                base_pos = strategyData.my_desired_position.copy()
                
                # Shift formation towards ball slightly (but don't chase)
                ball_influence = (ball_pos - base_pos) * 0.2  # 20% shift towards ball
                adjusted_pos = base_pos + ball_influence
                
                # Keep attackers in attacking half
                adjusted_pos[0] = max(adjusted_pos[0], -2.0)
                
                strategyData.my_desired_position = adjusted_pos
                drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.green, f"move_{my_unum}")
                
                # Face the ball while holding position
                return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)


    #--------------------------------------- Fat proxy auxiliary methods

    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg(self.kick_direction - r.imu_torso_orientation):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3)
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True)
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
                orientation = M.normalize_deg(orientation - r.imu_torso_orientation)
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")
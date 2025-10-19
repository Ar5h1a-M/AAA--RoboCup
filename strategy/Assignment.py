
# Step 1: Calculate Euclidean distances and create preference lists
 # Create preference lists - bots
 # Sort by distance
 # Create preference lists - roles
 #'            '
 
 
 # Step 2: Initialize all players + roles = un match
 
 # Step 3 & 4: Proposals until stable matching
 # Check if player has use up all preferences
  # Player proposes to next preferred role
  #evaluates the proposal- if role open = match,role been matched = comp prefs
   # Find player the role prefers- if prefers new player= unmatch current, Role prefers current match= reject new 
   # Step 5: Convert to output format 

import numpy as np

def role_assignment(teammate_positions, formation_positions):

    n = len(teammate_positions)
   

    def euclid_dist(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    

   
    player_preferences = {}
    for player_idex in range(n):
        distances = []
        for role_index in range(n):
            dist = euclid_dist(teammate_positions[player_idex],
                                     formation_positions[role_index])
            distances.append((dist, role_index))

        distances.sort()
        player_preferences[player_idex] = [role_index for dist, role_index in distances]
   
    roles_preferences = {}
    for role_index in range(n):
        distances = []
        for player_idex in range(n):
            dist = euclid_dist(formation_positions[role_index],
                                     teammate_positions[player_idex])
            distances.append((dist, player_idex))

        distances.sort()
        roles_preferences[role_index] = [player_idex for dist, player_idex in distances]

    unmatched_players = list(range(n))
    current_matches = {role_index: None for role_index in range(n)}
    proposal_index = {player_idex: 0 for player_idex in range(n)}  

    while unmatched_players:
        player_idex = unmatched_players.pop(0)
       

        if proposal_index[player_idex] >= n:
            continue
       

        role_index = player_preferences[player_idex][proposal_index[player_idex]]
        proposal_index[player_idex] += 1
       

        if current_matches[role_index] is None:

            current_matches[role_index] = player_idex
        else:

            current_match = current_matches[role_index]
            current_match_rank = roles_preferences[role_index].index(current_match)
            new_proposer_rank = roles_preferences[role_index].index(player_idex)
           
            if new_proposer_rank < current_match_rank:
                current_matches[role_index] = player_idex
                unmatched_players.append(current_match)  
            else:
                
                unmatched_players.append(player_idex)  
    point_preferences = {}
    for role_index, player_idex in current_matches.items():
        if player_idex is not None:
            unum = player_idex + 1  
            point_preferences[unum] = formation_positions[role_index]
   
    return point_preferences
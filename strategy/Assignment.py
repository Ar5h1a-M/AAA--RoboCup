import numpy as np

def role_assignment(teammate_positions, formation_positions):
    point_preferences = {}
    n = len(teammate_positions)
    
    #player + pos prefs using euclid(Arika)
    player_prefs = {}
    for i in range(n):
        dists = []
        for j in range(n):
            dx = teammate_positions[i][0] - formation_positions[j][0]
            dy = teammate_positions[i][1] - formation_positions[j][1]
            dists.append((np.sqrt(dx*dx + dy*dy), j))
        dists.sort()
        player_prefs[i] = [pos for _, pos in dists]
    
    position_prefs = {}
    for j in range(n):
        dists = []
        for i in range(n):
            dx = formation_positions[j][0] - teammate_positions[i][0]
            dy = formation_positions[j][1] - teammate_positions[i][1]
            dists.append((np.sqrt(dx*dx + dy*dy), i))
        dists.sort()
        position_prefs[j] = [player for _, player in dists]
    
    #all unmatched + preposal loop (Arshia)
    free_players = list(range(n))
    position_matches = [None] * n
    proposal_count = [0] * n
    
    while free_players:
        player = free_players[0]
        position = player_prefs[player][proposal_count[player]]
        proposal_count[player] += 1
        
        current_occupant = position_matches[position]
        
        if current_occupant is None:
            position_matches[position] = player
            free_players.pop(0)
        else:
            current_rank = position_prefs[position].index(current_occupant)
            new_rank = position_prefs[position].index(player)
            
            if new_rank < current_rank:
                position_matches[position] = player
                free_players.pop(0)
                free_players.insert(0, current_occupant)
    
    
    for pos, player in enumerate(position_matches):
        point_preferences[player + 1] = formation_positions[pos]
    
    return point_preferences

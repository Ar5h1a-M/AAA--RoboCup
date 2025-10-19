import numpy as np

def role_assignment(teammate_positions, formation_positions):

    n = len(teammate_positions)
   
#calc euclid distance
    def euclid_dist(x, y):
        return np.sqrt(np.sum((x - y)**2))
    

   #make prefs list 
    player_prefs = {}
    for p_index in range(n):
        dists = []
        for r_index in range(n):
            dist = euclid_dist(teammate_positions[p_index],
                                     formation_positions[r_index])
            dists.append((dist, r_index))
        player_prefs[p_index] = [r_index for dist, r_index in sorted(dists)]
   
    roles_prefs = {}
    for r_index in range(n):
        dists = []
        for p_index in range(n):
            dist = euclid_dist(formation_positions[r_index],
                                     teammate_positions[p_index])
            dists.append((dist, p_index))
        roles_prefs[r_index] = [p_index for dist, p_index in sorted(dists)]

    roles_prefs_rank = {
    r: {p: i for i, p in enumerate(pref_list)}
    for r, pref_list in roles_prefs.items()
}

#all unmatched at start 
    unmatched = list(range(n))
    matches = {r_index: None for r_index in range(n)}
    proposal_index = {p_index: 0 for p_index in range(n)}  
#keep proposing til matched stabley 
    while unmatched:
        p_index = unmatched.pop(0)

        if proposal_index[p_index] >= n:
            continue # this player tried everyone already
       

        r_index = player_prefs[p_index][proposal_index[p_index]]
        proposal_index[p_index] += 1
       

        if matches[r_index] is None:
#free role = match
            matches[r_index] = p_index
        else:
            #role taken compares prefs
            curr_match = matches[r_index]
            curr_match_rank = roles_prefs_rank[r_index][curr_match]
            new_proposer_rank = roles_prefs_rank[r_index][p_index]
           
            if new_proposer_rank < curr_match_rank:
                # role prefers new player
                matches[r_index] = p_index
                unmatched.append(curr_match)  
            else:
                 # rejected 
                unmatched.append(p_index)  
    point_prefs = {}
    for r_index, p_index in matches.items():
        if p_index is not None:
            unum = p_index + 1  
            point_prefs[unum] = formation_positions[r_index]
   
    return point_prefs
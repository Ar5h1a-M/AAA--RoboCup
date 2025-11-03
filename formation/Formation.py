import numpy as np

def GenerateBasicFormation():
    #have 2 defenders, 3 attackers to shoot at goal 
    formation = np.array([
        [-14.0, 0.0],    
        [-8.0, 0.0],     
        [2.0, -3.0],     
        [2.0, 3.0],       
        [5.0, 0.0]       
    ])
    
    return formation
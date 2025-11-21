#Problema1
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('O', 'H'), 
    ('O', 'W'),
    ('H', 'R'),
    ('W', 'R'),
    ('R', 'E'), 
    ('R', 'C')
])

cpd_o= TabularCPD(variable='O', variable_card=2,
                   values=[[0.3], 
                           [0.7]],
                   state_names={'O': ['cold', 'mild']})
cpd_h= TabularCPD(variable='H', variable_card=2,
                   values=[[0.9, 0.2],    
                           [0.1, 0.8]],   
                   evidence=['O'], evidence_card=[2],
                   state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']})

cpd_w= TabularCPD(variable='W', variable_card=2,
                   values=[[0.1, 0.6],    
                           [0.9, 0.4]],   
                   evidence=['O'], evidence_card=[2],
                   state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']})

cpd_r= TabularCPD(variable='R', variable_card=2,
                   values=[[0.6, 0.9, 0.3, 0.5],    
                           [0.4, 0.1, 0.7, 0.5]],   
                   evidence=['H', 'W'], evidence_card=[2, 2],
                   state_names={'R': ['warm', 'cool'], 'H': ['yes', 'no'], 'W': ['yes', 'no']})
cpd_e= TabularCPD(variable='E', variable_card=2,
                   values=[[0.8, 0.2],    
                           [0.2, 0.8]],   
                   evidence=['R'], evidence_card=[2],
                   state_names={'E': ['high', 'low'], 'R': ['warm', 'cool']})

cpd_c= TabularCPD(variable='C', variable_card=2,
                   values=[[0.85, 0.40],  
                           [0.15, 0.60]], 
                   evidence=['R'], evidence_card=[2],
                   state_names={'C': ['comfortable', 'uncomfortable'], 'R': ['warm', 'cool']})

model.add_cpds(cpd_o,cpd_h,cpd_w,cpd_r,cpd_e,cpd_c)

if model.check_model():
    infer = VariableElimination(model)
    #b) 
    #i
    query_h_c= infer.query(variables=['H'],evidence={'C': 'comfortable'})
    prob_h_yes_given_c= query_h_c.values[query_h_c.state_names['H'].index('yes')]
    
    print(f"P(H=yes | C=comfortable): {prob_h_yes_given_c:.4f}")

    #ii
    query_e_c= infer.query(variables=['E'], 
                            evidence={'C': 'comfortable'})
    prob_e_high_given_c= query_e_c.values[query_e_c.state_names['E'].index('high')]

    print(f"P(E=high | C=comfortable): {prob_e_high_given_c:.4f}")

    #iii
    map_estimate = infer.map_query(variables=['H', 'W'], evidence={'C': 'comfortable'})
    
    print(f"MAP estimate for (H, W) given C=comfortable: {map_estimate}")

    query_hw_c= infer.query(variables=['H', 'W'], 
                             evidence={'C': 'comfortable'})
    h_idx = query_hw_c.state_names['H'].index('yes')
    w_idx = query_hw_c.state_names['W'].index('no')
    prob_map_state = query_hw_c.values[1] 

    print(f"P(H=yes, W=no | C=comfortable) [MAP Prob]: {query_hw_c.values[1]}")
else:
    print("Model construction failed")

    
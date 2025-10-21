Model Setup_A1 considerations:

(1) Structural connectivity (A)- weights are derived by a standard exponential distribution (following the real-world structural data). 

    Re-arrange matrix based on weighted degree from highest to the lowest.    
    Normalization: $$\max\{\sum_{j=1}^N a_{ij}\} = 1 \forall i \in [N]$$

(2) K_{glob} \in [-1, 1]

(3) Modulation: Node with highest weighted degree

(4) Heterogeneity: setting all the nodes in the network to the mean heterogeneity vector

(5) Initialization: 10 connectivity matrix + 10 initialization points
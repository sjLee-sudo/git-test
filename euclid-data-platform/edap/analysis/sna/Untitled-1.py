

# Create a two-mode network from a document-word matrix
import networkx as nx
import numpy as np

# Create a document-word matrix (each row is a document, each column is a word)
doc_word = np.array([[1, 0, 0, 1], 
                     [0, 1, 1, 0], 
                     [1, 1, 0, 0]]) 

# Create an empty graph 
G = nx.Graph() 
  
# Add nodes with the node attribute "bipartite" 
G.add_nodes_from(['d1', 'd2', 'd3'], bipartite=0) 
G.add_nodes_from(['w1', 'w2', 'w3', 'w4'], bipartite=1) 

 # Add edges only between nodes of opposite node sets  
for i in range(3): # loop over documents  
    for j in range(4): # loop over words  

        # if the (i, j) entry in the doc-word matrix is nonzero  

        if doc_word[i][j] != 0:  

            # add an edge between document i and word j  

            G.add_edge('d' + str(i+1), 'w' + str(j+1)) 

            # add an edge weight equal to the (i, j) entry in the doc-word matrix  

            G['d' + str(i+1)]['w' + str(j+1)]['weight'] = doc_word[i][j]
            
print(G.edges)
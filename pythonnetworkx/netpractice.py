import networkx as nx
G = nx.Graph()

G.add_nodes_from([
    (4, {"color": "red"}),
    (5, {"color": "green"}),
])

print(G)
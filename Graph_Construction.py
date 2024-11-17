import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

# Define graph edges as vertex pairs
adj = np.array([[0,0,0,0,0,1,2,2,3,3,3,3,3,3,4,4,4,5,6,7,7,9,9,10,10,10,11,12,12,13,14,14,15,15,15,16,16,16,17,17,17,18,18,18,19,20,20,21,22,22,23,25,26,26,27,28,28,29,29,30,31,32,34,1,2,3,4,5,33,31,33,8,9,10,14,17,31,7,8,12,6,7,8,35,12,13,11,12,13,13,14,34,14,15,23,16,19,20,17,19,20,18,25,27,19,21,25,20,21,22,22,23,24,24,26,27,28,28,29,30,30,32,31,33,33,35],
                [1,2,3,4,5,33,31,33,8,9,10,14,17,31,7,8,12,6,7,8,35,12,13,11,12,13,13,14,34,14,15,23,16,19,20,17,19,20,18,25,27,19,21,25,20,21,22,22,23,24,24,26,27,28,28,29,30,30,32,31,33,33,35,0,0,0,0,0,1,2,2,3,3,3,3,3,3,4,4,4,5,6,7,7,9,9,10,10,10,11,12,12,13,14,14,15,15,15,16,16,16,17,17,17,18,18,18,19,20,20,21,22,22,23,25,26,26,27,28,28,29,29,30,31,32,34]])

# Determine the number of vertices
num_vertices = np.max(adj) + 1

# Initialize adjacency matrix with zeros
adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
A1 = adj_matrix

# Update adjacency matrix based on given edges
for i in range(adj.shape[1]):
    vertex1, vertex2 = adj[0, i], adj[1, i]
    A1[vertex1, vertex2] = 1
    A1[vertex2, vertex1] = 1  # Uncomment this line if the graph is undirected

# Create a graph
G = nx.Graph()

# Add vertices to the graph
num_nodes = adj.max()
G.add_nodes_from(range(1, num_nodes + 1))

# Add edges to the graph
for i in range(adj.shape[1]):
    node1 = adj[0, i]
    node2 = adj[1, i]
    G.add_edge(node1, node2)

print(G)
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Graph Visualization")
plt.show()

adj_matrix = nx.to_numpy_array(G)

# Function to find all paths between source and target of a specific length
def find_paths(graph, source, target, path, length):
    path = path + [source]
    if length == 0 and source == target:
        return [path]
    if length <= 0:
        return []
    paths = []
    for neighbor, weight in enumerate(graph[source]):
        if weight > 0 and neighbor not in path:
            new_paths = find_paths(graph, neighbor, target, path, length - 1)
            for new_path in new_paths:
                paths.append(new_path)
    return paths

# Function to find the diameter of an undirected graph
def graph_diameter(adj_matrix):
    num_vertices = len(adj_matrix)
    max_diameter = 0

    # Helper function to perform BFS from a starting vertex
    def bfs(start):
        visited = [False] * num_vertices
        distance = [0] * num_vertices

        queue = deque()
        queue.append(start)
        visited[start] = True

        while queue:
            vertex = queue.popleft()
            for neighbor, weight in enumerate(adj_matrix[vertex]):
                if weight > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    distance[neighbor] = distance[vertex] + 1
                    queue.append(neighbor)

        return max(distance)

    # Iterate through all vertices as starting points for BFS
    for vertex in range(num_vertices):
        diameter = bfs(vertex)
        if diameter > max_diameter:
            max_diameter = diameter

    return max_diameter

# Example adjacency matrix
adjacency_matrix = A1

diameter = graph_diameter(adjacency_matrix)
print(f"The graph diameter is {diameter}.")

# Initialize set to store path counts
path_counts = {}

# Initialize array to store column sums in num_paths_matrix
total_depth = np.zeros(len(adjacency_matrix))

# Loop through all desired distances from 1 to the diameter
for desired_distance in range(1, diameter + 1):
    for start_vertex in range(len(adjacency_matrix)):
        for end_vertex in range(len(adjacency_matrix)):
            if start_vertex != end_vertex:
                # Function to find the shortest distance between two nodes using Dijkstra's algorithm
                def shortest_distance(adj_matrix, start_vertex, end_vertex):
                    num_vertices = len(adj_matrix)
                    distances = [float('inf')] * num_vertices
                    visited = [False] * num_vertices
                    distances[start_vertex] = 0

                    for _ in range(num_vertices):
                        min_distance = float('inf')
                        min_vertex = -1
                        for v in range(num_vertices):
                            if not visited[v] and distances[v] < min_distance:
                                min_distance = distances[v]
                                min_vertex = v

                        if min_vertex == -1:
                            break

                        visited[min_vertex] = True

                        for v in range(num_vertices):
                            if not visited[v] and adj_matrix[min_vertex][v] > 0:
                                new_distance = distances[min_vertex] + adj_matrix[min_vertex][v]
                                if new_distance < distances[v]:
                                    distances[v] = new_distance

                    return distances[end_vertex]

                # Find the shortest path length between start and end nodes
                shortest_path_length = shortest_distance(adjacency_matrix, start_vertex, end_vertex)

                # Find all paths of desired distance from start to end nodes
                paths = find_paths(adjacency_matrix, start_vertex, end_vertex, [], desired_distance)

                # Count the number of paths
                num_paths = len(paths)

                # Print paths and verify if the desired distance matches the shortest path length
                if desired_distance == shortest_path_length:
                    print(f"All paths with length {desired_distance} from vertex {start_vertex} to vertex {end_vertex}:")
                    for path in paths:
                        print(path)
                    print(f"Number of paths: {num_paths}")
                    num_paths_matrix = np.zeros((len(adjacency_matrix), len(adjacency_matrix)))
                    # Store the number of paths in num_paths_matrix
                    num_paths_matrix[start_vertex][end_vertex] += num_paths
                    num_paths_matrix[start_vertex][end_vertex] *= desired_distance  # New addition
                    res = [sum(idx) for idx in zip(*num_paths_matrix)]
                    print("Sum of all values in the num_paths_matrix:")
                    print(res)
                    # Add column sums to total_depth
                    total_depth += np.sum(num_paths_matrix, axis=0)

# Print column sums in total_depth
print("Total depth for each vertex:")
print(total_depth)

# Calculate Mean Depth
mean_depth = np.zeros(len(adjacency_matrix))
mean_depth += total_depth / (num_vertices - 1)
print("Mean depth for each vertex:")
print(mean_depth)

# Calculate RA
RA = np.zeros(len(adjacency_matrix))
RA += 2 * (mean_depth - 1) / (num_vertices - 2)
print("Relative asymmetry for each vertex:")
print(RA)

# Calculate GL
GL = np.zeros(len(adjacency_matrix))
GL += 2 * (num_vertices * math.sqrt(num_vertices) - (2 * num_vertices) + 1) / ((num_vertices - 1) * (num_vertices - 2))

# Calculate RRA
RRA = np.zeros(len(adjacency_matrix))
RRA += RA / GL
print("Matrix RRA for each vertex:")
print(RRA)

# Draw graph with RRA values as labels
# Find vertex with the smallest RRA value
G.add_nodes_from(range(len(RRA)-1))
smallest_RRA_vertex = np.argmin(RRA)-1
node_labels = {vertex: f"Vertex {vertex}\nRRA: {RRA_value:.2f}" for vertex, RRA_value in enumerate(RRA)}
node_colors = ['lightblue' if i != smallest_RRA_vertex else 'red' for i in range(len(G.nodes))]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color='gray', node_size=2000)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')
plt.title("Graph with RRA Values")
plt.show()

########################## VERSION 1 #########################

# Draw the graph with RRA values as labels
# Find the vertex with the smallest RRA value
G.add_nodes_from(range(len(RRA) - 1))  # TRY EDITING THIS TO -1
smallest_RRA_vertex = np.argmin(RRA) - 1
node_labels = {vertex: f"Vertex {vertex}\nRRA: {RRA_value:.2f}" for vertex, RRA_value in enumerate(RRA)}
node_colors = ['lightblue' if i != smallest_RRA_vertex else 'red' for i in range(len(RRA))]
nx.draw(G, with_labels=True, node_color=node_colors, edge_color='gray', labels=node_labels)
plt.title("Graph Visualization")
plt.show()

# Initialize lists to store RRA values and their coordinates
RRA_values = []
RRA_pairs = []

# Calculate RRA for each vertex
for vertex in range(len(adjacency_matrix)):
    # Calculate RRA for the current vertex
    RRA_value = RRA[vertex]

    # Store the RRA value and corresponding coordinates in the list
    RRA_values.append(RRA_value)
    RRA_pair = (vertex, np.argmax(adjacency_matrix[vertex]))
    RRA_pairs.append(RRA_pair)

# Sort RRA pairs based on their values from smallest to largest
sorted_RRA_pairs = sorted(zip(RRA_values, RRA_pairs))

# Extract the sorted RRA values and corresponding vertex indices
sorted_RRA_values, sorted_RRA_vertex_pairs = zip(*sorted_RRA_pairs)

# Print the sorted RRA values and their corresponding vertices
print("RRA values sorted from smallest:")
for RRA_value, (vertex1, vertex2) in zip(sorted_RRA_values, sorted_RRA_vertex_pairs):
    print(f"RRA: {RRA_value}, Vertex: {vertex1}")

# Intelligibility analysis
# Calculation for RA
sum_RA = 0
sum_RA += sum(RA)
average_RA = 0
average_RA += sum_RA / num_vertices

x = np.zeros(len(adjacency_matrix))
x += RA - average_RA
x2 = np.zeros(len(adjacency_matrix))
x2 += x * x

sum_x2 = 0
sum_x2 += sum(x2)

# Calculation for RRA
sum_RRA = 0
sum_RRA += sum(RRA)
average_RRA = 0
average_RRA += sum_RRA / num_vertices

y = np.zeros(len(adjacency_matrix))
y += RRA - average_RRA
y2 = np.zeros(len(adjacency_matrix))
y2 += y * y

sum_y2 = 0
sum_y2 += sum(y2)

x2y2 = 0
x2y2 += sum_x2 * sum_y2

xy = np.zeros(len(adjacency_matrix))
xy += x * y
sum_xy = 0
sum_xy += sum(xy)

R_calculated = 0
R_calculated += sum_xy / math.sqrt(x2y2)
print("The value of R calculated is:")
print(R_calculated)

# Determine decision based on the correlation coefficient value
if R_calculated > 0.7:
    decision = "Strong positive correlation between RA and RRA. When one variable increases, the other tends to increase as well."
elif R_calculated < -0.7:
    decision = "Strong negative correlation between RA and RRA. When one variable increases, the other tends to decrease."
else:
    decision = "Weak or no correlation between RA and RRA. Changes in one variable do not strongly indicate changes in the other variable."

# Display the decision
print("Decision:", decision)



# Packages installation
"""

import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

!pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-geometric-temporal
!pip install torch_geometric==2.3.1
!pip install torch-geometric-temporal --upgrade
!pip install networkx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

"""Pre-processing Data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('BWI6.xlsx', header=None)

data = data.drop([0], axis=1) #column
data = data.drop([0], axis=0)



data #print dataset

"""Convert to Numpyarray, Pemisahan FItur sesuai Index, Normalisasi"""

#Convert dataset to numpyarray
data = np.array(data)

# Assuming each feature corresponds to specific columns (nodes)
CH = data[:, 0::8]    # Example: CH data from every 8th column
T = data[:, 1::8]     # Example: T data from every 8th column
WS = data[:, 2::8]    # Example: WS data from every 8th column
HD = data[:, 3::8]    # Example: HD data from every 8th column
PV_CR = data[:, 4::8] # Example: PV_CR data from every 8th column
K_CR = data[:, 5::8]  # Example: K_CR data from every 8th column
PV_CB = data[:, 6::8] # Example: PV_CB data from every 8th column
K_CB = data[:, 7::8]  # Example: K_CB data from every 8th column

periode = np.arange(0,28)

def normalize(feature):
    normalized_feature = np.zeros_like(feature)
    for i in range(feature.shape[1]):  # Iterasi setiap kolom (node)
        column = feature[:, i]   # Pilih setiap kolom (node)
        col_min = np.min(column)     # Hitung nilai minimum untuk kolom tersebut
        col_max = np.max(column)     # Hitung nilai maksimum untuk kolom tersebut
        normalized_feature[:, i] = 0.1 + (((column - col_min) * (0.9 - 0.1)) / (col_max - col_min))  # Normalisasi setiap kolom (node)
    return normalized_feature


# Normalize per column (per node) within the CH feature
CH = normalize(CH)
T = normalize(T)
WS = normalize(WS)
HD= normalize(HD)
PV_CR = normalize(PV_CR)
K_CR = normalize(K_CR)
PV_CB = normalize(PV_CB)
K_CB = normalize(K_CB)

"""# Graph Visualization"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is already defined
periode_data = data

adj = np.array([[0, 0, 0, 0, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 10, 10, 10, 11, 11, 12, 12, 13, 14, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 20, 20, 20, 21, 22, 23],
                 [1, 4, 5, 6, 3, 4, 22, 5, 19, 22, 6, 18, 19, 20, 7, 15, 18, 8, 14, 15, 10, 12, 14, 10, 11, 12, 13, 12, 13, 13, 14, 14, 15, 16, 17, 16, 18, 17, 18, 18, 21, 20, 21, 20, 22, 23, 21, 23, 24, 24, 23, 24]])

# Create a graph
G = nx.Graph()

# Add nodes to the graph
num_nodes = adj.max()
G.add_nodes_from(range(0, num_nodes + 1))

# Add edges to the graph
for i in range(adj.shape[1]):
    node1 = adj[0, i]
    node2 = adj[1, i]
    G.add_edge(node1, node2)

# Print the graph
print(G)
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Street Graph Visualization")
plt.show()

adj_matrix = nx.to_numpy_array(G)

# Convert the adjacency matrix to a pandas DataFrame
adj_matrix = pd.DataFrame(adj_matrix, columns=[f'Node {i}' for i in range(adj_matrix.shape[1])],
                          index=[f'Node {i}' for i in range(adj_matrix.shape[0])])

# Create a heatmap visualization of the adjacency matrix
plt.figure(figsize=(10, 10))  # Set the figure size
plt.imshow(adj_matrix, cmap='hot', interpolation='nearest')

# Set axis labels
plt.xlabel('Nodes')
plt.ylabel('Nodes')

# Show the colorbar for reference
plt.colorbar(ticks=[0, 1])

# Add the text (0s and 1s) to the heatmap with conditional colors
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        color = 'black' if adj_matrix.iloc[i, j] == 1 else 'white'
        plt.text(j, i, int(adj_matrix.iloc[i, j]), ha='center', va='center', color=color)

# Set the ticks and tick labels
node_labels = [f'Node {i}' for i in range(adj_matrix.shape[0])]
plt.xticks(ticks=np.arange(adj_matrix.shape[1]), labels=node_labels, rotation=90)
plt.yticks(ticks=np.arange(adj_matrix.shape[0]), labels=node_labels)

# Display the plot
plt.title('Street Graph Adjacency Matrix')
plt.show()

adj_matrix

"""# Dataset Preparation
Variable lags adalah window size
"""

dataset = np.stack((CH, T, WS, HD, PV_CR, K_CR), axis=1)
dataset = np.transpose(dataset, (2,0,1))
lags = 4

def get_targets_and_features(lags, dataset):
    stacked_target = dataset
    features = [
        stacked_target[i : i + lags, :, :]
        for i in range(stacked_target.shape[0] - lags)
    ]
    targets = [
        stacked_target[i + lags, :, :]
        for i in range(stacked_target.shape[0] - lags)  # stacked_target.shape[0] = 28 dg lags 3, sehingga (for i in range(stacked_target.shape[0] - lags) = 25, krn apabila i=26 it has 29 data samples (invalid)
    ]
    features = np.transpose(features, (0, 2, 3, 1)) #layer RNN
    return features, targets

features, targets = get_targets_and_features(lags, dataset)

features=np.array(features).astype(float)
targets=np.array(targets).astype(float)

print("Dataset Shape:", dataset.shape) # output 28, 25, 5. 28 time steps, 25 nodes, and 5 features per node in your dataset.
print("Lags:", lags) #3 is the 3 previous time steps to predict the target at the next time step.
print("features Shape:", features.shape) # only create features for the first 25 time steps (line 9), 25 = the number of nodes, 5 = the number of features per node, The fourth dimension (3) represents the lagged time steps.
print("targets Shape:", targets.shape) # same with features but tidak punya lags karena hanya berisi current step

"""## Split Train and Test Dataset
Train ratio = 60%
Test ratio = 40%
"""

from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal
import torch

adj = adj
edge_weight = np.ones(adj.shape[1]) #edge weight untuk mencari number of edges
features= np.array(features)
targets=np.array(targets)

#StaticGraphTemporalSignal untuk menangani graph static
#edge_index seperti adj
#edge_weight jumlah sisi/edges yang saling terhubung
loader =  StaticGraphTemporalSignal(edge_index = adj,  edge_weight = edge_weight, features = features, targets = targets)
train_dataset, test_dataset = temporal_signal_split(loader, train_ratio=0.5)

edge_weight

"""# Model Declaration
TGCN
"""

# Prediction Model Construction

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.periods =4
        self.tconv = TGCN(node_features,32)
        self.linear = torch.nn.Linear(32, node_features)

    def forward(self, x, edge_index, edge_weight): #input data diteruskan untuk menghasilkan output prediksi
        if edge_index.shape[0] == 0:
            # Handle empty graphs (e.g., skip processing or return default values)
            return None
        H = self.tconv(x[:, :, 0], edge_index, edge_weight)
        for period in range(self.periods-1):
            H = self.tconv(x[:, :, period+1], edge_index, edge_weight, H)
        H = F.relu(H)
        H = self.linear(H)
        return H

"""# Training Process"""

from tqdm import tqdm

device = torch.device("cpu")
model = RecurrentGCN(node_features = 6).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
training_cost = []
for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        snapshot.to(device)
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    if epoch % 20 == 0  :
        print(f"Current epoch:{epoch}, current training loss:{cost}")
        current_cost = cost
        training_cost.append(current_cost.cpu().detach().numpy())
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

"""# Training Loss Visualization"""

# Create a list of epochs
epochs = np.arange(1, len(training_cost) + 1)
epochs = epochs*20

# Create a list of losses
losses = training_cost

# Plot the loss vs. epoch
plt.plot(epochs, losses, 'r--')

# Add a title
plt.title('Loss vs. Epoch')

# Add labels to the x-axis and y-axis
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Add a scale every 20 epochs
plt.xticks(epochs)

# Show the plot
plt.show()

"""# Testing Process"""

model.eval()
cost = 0
results = []
ground_truth = []

for time, snapshot in enumerate(train_dataset):
    ground_truth.append(snapshot.y.cpu().detach().numpy())

for time, snapshot in enumerate(test_dataset):
    snapshot.to(device)
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    results.append(y_hat.cpu().detach().numpy())
    ground_truth.append(snapshot.y.cpu().detach().numpy())
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))


results = np.array(results)
results = np.transpose(results, (1, 2, 0))

ground_truth = np.array(ground_truth)
ground_truth = np.transpose(ground_truth, (1, 2, 0))

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to evaluate the model
def evaluate_model(model, dataset, device):
    model.eval()
    predictions = []
    ground_truth = []
    total_cost = 0
    num_samples = 0

    for snapshot in dataset:
        snapshot.to(device)
        with torch.no_grad():
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            predictions.append(y_hat.cpu().numpy())
            ground_truth.append(snapshot.y.cpu().numpy())
            total_cost += torch.mean((y_hat - snapshot.y) ** 2).item()
            num_samples += 1

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    mse = mean_squared_error(ground_truth, predictions)
    rmse_val = rmse(ground_truth, predictions)
    mae_val = mean_absolute_error(ground_truth, predictions)
    r2_val = r2_score(ground_truth, predictions)

    return mse, rmse_val, mae_val, r2_val

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate on test data
test_mse, test_rmse, test_mae, test_r2 = evaluate_model(model, test_dataset, device)
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2: {test_r2:.4f}")

"""# Forecasting Visualization"""

def plot_results(periode, results, truth, title, xlabel, ylabel, ax):
    ax.plot(periode, truth, linewidth=5, label="Ground Truth")
    ax.plot(periode[-12:], results, '--', linewidth=5, label="Forecasting Result")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

CH_res = results[21,0,:] #predicted value
T_res = results[21,1,:]
WS_res = results[21,2,:]
HD_res = results[21,3,:]
PV_CB_res = results[21,4,:]
K_CB_res = results[21,5,:]

CH_truth = ground_truth[21,0,:] #actual value
T_truth = ground_truth[21,1,:]
WS_truth = ground_truth[21,2,:]
HD_truth = ground_truth[21,3,:]
PV_CB_truth = ground_truth[21,4,:]
K_CB_truth = ground_truth[21,5,:]

periode = np.arange(0, 24)
# Define the subplots layout and increase the size of the whole figure
fig, axes = plt.subplots(3, 2, figsize=(20, 20))

plot_results(periode,CH_res, CH_truth,'Prediksi Curah Hujan', 'Periode', 'Curah Hujan', axes[0, 0])
plot_results(periode,T_res, T_truth ,'Prediksi Suhu', 'Periode', 'Suhu', axes[0, 1])
plot_results(periode, WS_res, WS_truth, 'Prediksi Kecepatan Angin', 'Periode', 'Kecepatan Angin', axes[1, 0])
plot_results(periode, HD_res, HD_truth, 'Prediksi Kelembaban Udara', 'Periode', 'Kelembaban Udara', axes[1, 1])
plot_results(periode, PV_CB_res, PV_CB_truth, 'Prediksi Produksi Cabai Besar', 'Periode', 'Produksi Cabai Besar', axes[2, 0])
plot_results(periode, K_CB_res, K_CB_truth, 'Prediksi Ketersediaan Cabai Besar', 'Periode', 'Ketersediaan Cabai Besar', axes[2, 1])

def plot_results(periode, results, truth, title, xlabel, ylabel, ax):
    ax.plot(periode, truth, linewidth=5, label="Ground Truth")
    ax.plot(periode[-12:], results, '--', linewidth=5, label="Forecasting Result")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

CH_res = results[22,0,:] #predicted value
T_res = results[22,1,:]
WS_res = results[22,2,:]
HD_res = results[22,3,:]
PV_CB_res = results[22,4,:]
K_CB_res = results[22,5,:]

CH_truth = ground_truth[22,0,:] #actual value
T_truth = ground_truth[22,1,:]
WS_truth = ground_truth[22,2,:]
HD_truth = ground_truth[22,3,:]
PV_CB_truth = ground_truth[22,4,:]
K_CB_truth = ground_truth[22,5,:]

periode = np.arange(0, 24)
# Define the subplots layout and increase the size of the whole figure
fig, axes = plt.subplots(3, 2, figsize=(20, 20))

plot_results(periode,CH_res, CH_truth,'Prediksi Curah Hujan', 'Periode', 'Curah Hujan', axes[0, 0])
plot_results(periode,T_res, T_truth ,'Prediksi Suhu', 'Periode', 'Suhu', axes[0, 1])
plot_results(periode, WS_res, WS_truth, 'Prediksi Kecepatan Angin', 'Periode', 'Kecepatan Angin', axes[1, 0])
plot_results(periode, HD_res, HD_truth, 'Prediksi Kelembaban Udara', 'Periode', 'Kelembaban Udara', axes[1, 1])
plot_results(periode, PV_CB_res, PV_CB_truth, 'Prediksi Produksi Cabai Besar', 'Periode', 'Produksi Cabai Besar', axes[2, 0])
plot_results(periode, K_CB_res, K_CB_truth, 'Prediksi Ketersediaan Cabai Besar', 'Periode', 'Ketersediaan Cabai Besar', axes[2, 1])

def plot_results(periode, results, truth, title, xlabel, ylabel, ax):
    ax.plot(periode, truth, linewidth=5, label="Ground Truth")
    ax.plot(periode[-10:], results, '--', linewidth=5, label="Forecasting Result")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

CH_res = results[23,0,:] #predicted value
T_res = results[23,1,:]
WS_res = results[23,2,:]
HD_res = results[23,3,:]
PV_CB_res = results[23,4,:]
K_CB_res = results[23,5,:]


CH_truth = ground_truth[23,0,:] #actual value
T_truth = ground_truth[23,1,:]
WS_truth = ground_truth[23,2,:]
HD_truth = ground_truth[23,3,:]
PV_CB_truth = ground_truth[23,4,:]
K_CB_truth = ground_truth[23,5,:]


periode = np.arange(0, 24)
# Define the subplots layout and increase the size of the whole figure
fig, axes = plt.subplots(3, 2, figsize=(20, 20))

plot_results(periode,CH_res, CH_truth,'Prediksi Curah Hujan', 'Periode', 'Curah Hujan', axes[0, 0])
plot_results(periode,T_res, T_truth ,'Prediksi Suhu', 'Periode', 'Suhu', axes[0, 1])
plot_results(periode, WS_res, WS_truth, 'Prediksi Kecepatan Angin', 'Periode', 'Kecepatan Angin', axes[1, 0])
plot_results(periode, HD_res, HD_truth, 'Prediksi Kelembaban Udara', 'Periode', 'Kelembaban Udara', axes[1, 1])
plot_results(periode, PV_CB_res, PV_CB_truth, 'Prediksi Produksi Cabai Besar', 'Periode', 'Produksi Cabai Besar', axes[2, 0])
plot_results(periode, K_CB_res, K_CB_truth, 'Prediksi Ketersediaan Cabai Besar', 'Periode', 'Ketersediaan Cabai Besar', axes[2, 1])

"""# Multi-step Forecasting
The forecasted values from one time step are used as input for predicting the subsequent time step, and this process can be repeated iteratively to forecast multiple future time steps.
"""

def get_features(lags, dataset):
    stacked_target = dataset
    features = stacked_target[stacked_target.shape[0] - lags : stacked_target.shape[0], :, :]

    features = np.array(features, dtype='float32')
    features = np.transpose(features, (1,2,0))
    features = torch.from_numpy(features)

    return features

input = get_features(lags, dataset)
print("input shape", input.shape)

"""# Run Multi-step"""

copied_dataset = np.copy(dataset)
model.eval()
step_ahead = 4

for i in range(step_ahead):
    y_hat = model(input, snapshot.edge_index, snapshot.edge_attr)
    copied_dataset = np.vstack((copied_dataset, [y_hat.cpu().detach().numpy()]))
    input = get_features(lags, copied_dataset)

print(copied_dataset.shape)
copied_dataset = np.transpose(copied_dataset, (1, 2, 0))

"""# Multi-step forecasting Dominator 0"""

def plot_results_multi(periode, result, title, xlabel, ylabel, ax):
    ax.plot(periode[:dataset.shape[0]], result[:dataset.shape[0]], linewidth=5, label="Ground Truth", color='blue')
    ax.plot(periode[-step_ahead - 1:], result[-step_ahead - 1:], '--', linewidth=5, label="Multi-step forecast", color='red')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

# Define the denormalization function
def denormalize(predicted_values, Xmin, Xmax):
    return (((predicted_values - 0.1) * (Xmax - Xmin)) / 0.8) + Xmin

# Extract predicted values
K_CB_res = copied_dataset[0, 5, :]
# K_CB_res = copied_dataset[0, 7, :]

# Calculate min and max values for normalization from copied_dataset

Xmin_CB = np.min(data[:, 7::8])
Xmax_CB = np.max(data[:, 7::8])

# Denormalize the results
K_CB_res_denormalized = denormalize(K_CB_res, Xmin_CB, Xmax_CB)

print({Xmin_CB})
print({Xmax_CB})


# Debugging prints with three decimal places
np.set_printoptions(precision=3)
print(f"Prediksi ketersediaan Cabai Besar (Normalisasi): {K_CB_res[28:36]}"))
print(f"Prediksi ketersediaan Cabai Besar (DeNormalisasi): {K_CB_res_denormalized[28:36]}")

# Define the periods for plotting
periode = np.arange(0, copied_dataset.shape[2])

num_plots = 1
# Define the subplots layout and increase the size of the whole figure
fig, ax = plt.subplots(1, 1, figsize=(30, 10))

# Plot the results
plot_results_multi(periode, K_CB_res, 'Prediksi Ketersediaan Cabai Rawit', 'Periode', 'Ketersediaan Cabai Besar', ax)

plt.show()

"""# Multi-step forecasting Dominator 3"""

def plot_results_multi(periode, result, title, xlabel, ylabel, ax):
    ax.plot(periode[:dataset.shape[0]], result[:dataset.shape[0]], linewidth=5, label="Ground Truth", color='blue')
    ax.plot(periode[-step_ahead - 1:], result[-step_ahead - 1:], '--', linewidth=5, label="Multi-step forecast", color='red')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

# Define the denormalization function
def denormalize(predicted_values, Xmin, Xmax):
    return (((predicted_values - 0.1) * (Xmax - Xmin)) / 0.8) + Xmin

# Extract predicted values
K_CR_res = copied_dataset[3, 5, :]
# K_CB_res = copied_dataset[3, 7, :]

# Calculate min and max values for normalization from copied_dataset
Xmin_CB = np.min(data[:, 7::8])
Xmax_CB = np.max(data[:, 7::8])

# Denormalize the results
K_CB_res_denormalized = denormalize(K_CB_res, Xmin_CB, Xmax_CB)

# Debugging prints with three decimal places
np.set_printoptions(precision=3)
print(f"Prediksi ketersediaan Cabai Besar (Normalisasi): {K_CB_res[28:36]}")
print(f"Prediksi ketersediaan Cabai Besar (DeNormalisasi): {K_CB_res_denormalized[28:36]}")

# Define the periods for plotting
periode = np.arange(0, copied_dataset.shape[2])

# Define the periods for plotting
periode = np.arange(0, copied_dataset.shape[2])

num_plots = 1
# Define the subplots layout and increase the size of the whole figure
fig, ax = plt.subplots(1, 1, figsize=(30, 10))

# Plot the results
plot_results_multi(periode, K_CB_res, 'Prediksi Ketersediaan Cabai Rawit', 'Periode', 'Ketersediaan Cabai Besar', ax)

plt.show()

"""# Multi-step forecasting Dominator 10"""

def plot_results_multi(periode, result, title, xlabel, ylabel, ax):
    ax.plot(periode[:dataset.shape[0]], result[:dataset.shape[0]], linewidth=5, label="Ground Truth", color='blue')
    ax.plot(periode[-step_ahead - 1:], result[-step_ahead - 1:], '--', linewidth=5, label="Multi-step forecast", color='red')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

# Define the denormalization function
def denormalize(predicted_values, Xmin, Xmax):
    return (((predicted_values - 0.1) * (Xmax - Xmin)) / 0.8) + Xmin

# Extract predicted values
K_CR_res = copied_dataset[10, 5, :]

# Calculate min and max values for normalization from copied_dataset
Xmin_CB = np.min(data[:, 7::8])
Xmax_CB = np.max(data[:, 7::8])

# Denormalize the results
K_CB_res_denormalized = denormalize(K_CB_res, Xmin_CB, Xmax_CB)

# Debugging prints with three decimal places
np.set_printoptions(precision=3)
print(f"Prediksi ketersediaan Cabai Besar (Normalisasi): {K_CB_res[28:36]}")
print(f"Prediksi ketersediaan Cabai Besar (DeNormalisasi): {K_CB_res_denormalized[28:36]}")

# Define the periods for plotting
periode = np.arange(0, copied_dataset.shape[2])

# Define the periods for plotting
periode = np.arange(0, copied_dataset.shape[2])

num_plots = 1
# Define the subplots layout and increase the size of the whole figure
fig, ax = plt.subplots(1, 1, figsize=(30, 10))

# Plot the results
plot_results_multi(periode, K_CB_res, 'Prediksi Ketersediaan Cabai Rawit', 'Periode', 'Ketersediaan Cabai Besar', ax)

plt.show()

"""# Multi-step forecasting Dominator 14"""

def plot_results_multi(periode, result, title, xlabel, ylabel, ax):
    ax.plot(periode[:dataset.shape[0]], result[:dataset.shape[0]], linewidth=5, label="Ground Truth", color='blue')
    ax.plot(periode[-step_ahead - 1:], result[-step_ahead - 1:], '--', linewidth=5, label="Multi-step forecast", color='red')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

# Define the denormalization function
def denormalize(predicted_values, Xmin, Xmax):
    return (((predicted_values - 0.1) * (Xmax - Xmin)) / 0.8) + Xmin

# Extract predicted values
K_CB_res = copied_dataset[14, 5, :]

# Calculate min and max values for normalization from copied_dataset
Xmin_CB = np.min(data[:, 7::8])
Xmax_CB = np.max(data[:, 7::8])

# Denormalize the results
K_CB_res_denormalized = denormalize(K_CB_res, Xmin_CB, Xmax_CB)

# Debugging prints with three decimal places
np.set_printoptions(precision=3)
print(f"Prediksi ketersediaan Cabai Besar (Normalisasi): {K_CB_res[28:36]}")
print(f"Prediksi ketersediaan Cabai Besar (DeNormalisasi): {K_CB_res_denormalized[28:36]}")

# Define the periods for plotting
periode = np.arange(0, copied_dataset.shape[2])

num_plots = 1
# Define the subplots layout and increase the size of the whole figure
fig, ax = plt.subplots(1, 1, figsize=(30, 10))

# Plot the results
plot_results_multi(periode, K_CB_res, 'Prediksi Ketersediaan Cabai Rawit', 'Periode', 'Ketersediaan Cabai Besar', ax)

plt.show()

"""# Multi-step forecasting Dominator 20"""

def plot_results_multi(periode, result, title, xlabel, ylabel, ax):
    ax.plot(periode[:dataset.shape[0]], result[:dataset.shape[0]], linewidth=5, label="Ground Truth", color='blue')
    ax.plot(periode[-step_ahead - 1:], result[-step_ahead - 1:], '--', linewidth=5, label="Multi-step forecast", color='red')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

# Define the denormalization function
def denormalize(predicted_values, Xmin, Xmax):
    return (((predicted_values - 0.1) * (Xmax - Xmin)) / 0.8) + Xmin

# Extract predicted values
K_CB_res = copied_dataset[20, 5, :]

# Calculate min and max values for normalization from copied_dataset
Xmin_CB = np.min(data[:, 7::8])
Xmax_CB = np.max(data[:, 7::8])

# Denormalize the results
K_CB_res_denormalized = denormalize(K_CB_res, Xmin_CB, Xmax_CB)

# Debugging prints with three decimal places
np.set_printoptions(precision=3)
print(f"Prediksi ketersediaan Cabai Besar (Normalisasi): {K_CB_res[28:36]}")
print(f"Prediksi ketersediaan Cabai Besar (DeNormalisasi): {K_CB_res_denormalized[28:36]}")

# Define the periods for plotting
periode = np.arange(0, copied_dataset.shape[2])

num_plots = 1
# Define the subplots layout and increase the size of the whole figure
fig, ax = plt.subplots(1, 1, figsize=(30, 10))

# Plot the results
plot_results_multi(periode, K_CB_res, 'Prediksi Ketersediaan Cabai Rawit', 'Periode', 'Ketersediaan Cabai Besar', ax)

plt.show()

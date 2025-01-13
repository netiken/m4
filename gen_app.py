import numpy as np
import json


def assign_clients_to_storage(matrix_size):
    """
    Given an n x n matrix, randomly assign 1/4 as clients and 3/4 as storage,
    then randomly generate traffic intensity from clients to storage.

    Parameters:
        matrix_size (int): Size of the square matrix (n).

    Returns:
        np.ndarray: New matrix with updated traffic values.
    """
    # Generate a matrix with zeros
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Flatten the matrix indices
    total_elements = matrix_size * matrix_size

    # Randomly select 1/4 indices as clients and 3/4 as storage
    indices = np.arange(matrix_size)
    np.random.shuffle(indices)

    client_count = matrix_size // 4

    # Split indices into clients and storage
    clients = indices[:client_count]
    storage = indices[client_count:]

    # Assign random traffic intensity only from clients to storage
    for i in range(len(clients) * len(storage)):
        client_idx = np.random.choice(clients)
        assigned_storage = np.random.choice(storage)
        traffic_intensity = np.random.randint(500, 1000)  # Random traffic intensity
        matrix[client_idx][assigned_storage] = traffic_intensity
    matrix += matrix.T
    return matrix


# Load JSON file
with open(
    "/data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_b_4_16.json",
    "r",
) as f:
    data = json.load(f)

# Read nr_racks as n
n = data["nr_racks"]

# Generate the data
new_matrix = assign_clients_to_storage(n)

# Replace inner matrix
data["matrix"]["inner"] = new_matrix.tolist()

# Save the updated JSON
with open(
    "/data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_d_4_16.json",
    "w",
) as f:
    json.dump(data, f, indent=2)

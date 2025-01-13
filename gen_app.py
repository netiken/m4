import numpy as np
import json


def assign_clients_to_storage(matrix_size):
    """
    Given an n x n matrix, randomly assign 1/4 as clients and 3/4 as storage,
    then randomly generate symmetric traffic intensity from clients to storage.

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
    indices = np.arange(total_elements)
    # np.random.shuffle(indices)

    client_count = total_elements // 4
    storage_count = total_elements - client_count

    client_indices = indices[:client_count]
    storage_indices = indices[client_count:]

    # Map the flat indices to 2D indices
    clients = np.unravel_index(client_indices, (matrix_size, matrix_size))
    storage = np.unravel_index(storage_indices, (matrix_size, matrix_size))

    # Assign random symmetric traffic intensity between clients and storage
    for client_idx in zip(*clients):
        storage_idx = np.random.choice(len(storage[0]))
        assigned_storage = (storage[0][storage_idx], storage[1][storage_idx])
        traffic_intensity = np.random.randint(100, 1000)  # Random traffic intensity
        matrix[client_idx] = traffic_intensity
        matrix[assigned_storage] = traffic_intensity

        # Ensure symmetry
        matrix[assigned_storage[0], assigned_storage[1]] = traffic_intensity
        matrix[client_idx[0], client_idx[1]] = traffic_intensity
    matrix += matrix.T
    return matrix


# Load JSON file
with open(
    "/data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_b_2_16.json",
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
    "/data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_d_2_16.json",
    "w",
) as f:
    json.dump(data, f, indent=2)

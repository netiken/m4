import random
import json
import numpy as np


# Function to upsample the inner matrix
def upsample_inner(original, new_rows, new_cols):
    """Upsamples the original matrix to the desired size."""
    scale_row = new_rows / len(original)
    scale_col = new_cols / len(original[0])
    row_factor = int(np.ceil(scale_row))
    col_factor = int(np.ceil(scale_col))
    return np.kron(original, np.ones((row_factor, col_factor))).astype(int).tolist()


# Generate unique idx2name values
def generate_idx2name(total_indices):
    """Generates unique idx2name values as a list."""
    idx2name = []
    used_names = set()

    def random_hex_name():
        while True:
            name = "".join(random.choices("abcdef0123456789", k=8))
            if name not in used_names:
                used_names.add(name)
                return name

    for _ in range(total_indices):
        idx2name.append(random_hex_name())

    return idx2name


input_file = "./parsimon-eval/workload/spatials/cluster_b.json"

output_file_str_list = [("_12k", 16), ("_24k", 32), ("_36k", 48), ("_48k", 64)]
nr_racks_per_pod = 48  # Example value

for output_file_str, nr_pods in output_file_str_list:
    output_file = f"./parsimon-eval/workload/spatials/cluster_b{output_file_str}.json"
    # Load the JSON file
    with open(input_file, "r") as file:
        data = json.load(file)

    # Configuration: Number of pods and racks per pod
    nr_racks = nr_pods * nr_racks_per_pod

    # Original data
    original_inner = data["matrix"]["inner"]
    original_pod2tors = data.get("pod2tors", {})
    original_nr_pods = data.get("nr_pods", len(original_pod2tors))
    original_nr_racks = data.get("nr_racks", len(original_inner))

    # Upsample the data
    upsampled_inner = upsample_inner(original_inner, nr_racks, nr_racks)

    # Generate new pod2tors mapping
    new_pod2tors = {}
    idx2name = generate_idx2name(nr_racks)

    idx_list = list(range(nr_racks))
    random.shuffle(idx_list)  # Shuffle to ensure random assignment

    for pod in range(nr_pods):
        pod_name = "".join(random.choices("abcdef0123456789", k=8))
        rack_indices = idx_list[pod * nr_racks_per_pod : (pod + 1) * nr_racks_per_pod]
        rack_names = [idx2name[idx] for idx in rack_indices]
        new_pod2tors[pod_name] = rack_names

    # Create the upsampled data
    upsampled_data = {
        "matrix": {
            "inner": upsampled_inner,
            "idx2name": idx2name,  # Now a list, following original format
        },
        "pod2tors": new_pod2tors,
        "nr_pods": nr_pods,
        "nr_racks": nr_racks,
    }

    # Save the upsampled data to a new file
    with open(output_file, "w") as file:
        json.dump(upsampled_data, file, indent=4)

    print(f"Upsampled data saved to {output_file}")

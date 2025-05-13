import random
import json
import numpy as np
import os


# Function to upsample the inner matrix
def upsample_inner(original, new_rows, new_cols):
    """Upsamples the original matrix to the desired size."""
    # Ensure we get exactly the target dimensions
    result = np.zeros((new_rows, new_cols), dtype=int)
    orig_rows, orig_cols = len(original), len(original[0])
    
    # Calculate scaling factors
    row_scale = new_rows / orig_rows
    col_scale = new_cols / orig_cols
    
    # Map each new position to original position
    for i in range(new_rows):
        for j in range(new_cols):
            orig_i = int(i / row_scale)
            orig_j = int(j / col_scale)
            result[i][j] = original[orig_i][orig_j]
    
    return result.tolist()


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


def sample_inner(original, target_rows, target_cols):
    """Samples the original matrix to match target dimensions."""
    # Create new matrix of target size
    result = np.zeros((target_rows, target_cols), dtype=int)
    orig_rows, orig_cols = len(original), len(original[0])
    
    # Calculate sampling ratios
    row_ratio = orig_rows / target_rows
    col_ratio = orig_cols / target_cols
    
    # Sample from original matrix
    for i in range(target_rows):
        for j in range(target_cols):
            # Map target position to original position
            orig_i = int(i * row_ratio)
            orig_j = int(j * col_ratio)
            result[i][j] = original[orig_i][orig_j]
    
    return result.tolist()


def repeat_inner(original, target_rows, target_cols):
    """Repeats the original matrix while preserving spatial distribution patterns."""
    # Create new matrix of target size
    result = np.zeros((target_rows, target_cols), dtype=int)
    orig_rows, orig_cols = len(original), len(original[0])
    
    # Calculate how many times we need to repeat in each dimension
    row_repeats = target_rows // orig_rows + (1 if target_rows % orig_rows else 0)
    col_repeats = target_cols // orig_cols + (1 if target_cols % orig_cols else 0)
    
    # Repeat the original pattern while maintaining spatial distribution
    for i in range(target_rows):
        for j in range(target_cols):
            # Map to original position while preserving relative position in the pattern
            orig_i = i % orig_rows
            orig_j = j % orig_cols
            result[i][j] = original[orig_i][orig_j]
    
    # If we have extra rows/cols, fill them by repeating the last row/col
    if target_rows % orig_rows:
        for i in range(orig_rows * (row_repeats - 1), target_rows):
            for j in range(target_cols):
                result[i][j] = result[i % orig_rows][j]
    
    if target_cols % orig_cols:
        for i in range(target_rows):
            for j in range(orig_cols * (col_repeats - 1), target_cols):
                result[i][j] = result[i][j % orig_cols]
    
    return result.tolist()


def scale_inner(original, target_rows, target_cols):
    """Scales the original matrix while preserving spatial distribution patterns."""
    # Create new matrix of target size
    result = np.zeros((target_rows, target_cols), dtype=int)
    orig_rows, orig_cols = len(original), len(original[0])
    
    # Calculate scaling factors
    row_scale = orig_rows / target_rows
    col_scale = orig_cols / target_cols
    
    # Scale the original pattern while preserving spatial distribution
    for i in range(target_rows):
        for j in range(target_cols):
            # Map target position back to original position
            orig_i = int(i * row_scale)
            orig_j = int(j * col_scale)
            result[i][j] = original[orig_i][orig_j]
    
    return result.tolist()


def upsample_cluster(input_file, output_file, target_nr_racks, target_nr_racks_per_pod):
    """Scale a cluster file to match the target number of racks while preserving spatial patterns."""
    # Load the input JSON file
    with open(input_file, "r") as file:
        data = json.load(file)

    # Validate input data structure
    if "matrix" not in data or "inner" not in data["matrix"]:
        raise ValueError(f"Invalid input file format in {input_file}: missing matrix.inner")
    
    # Get the target number of racks from the input file
    nr_racks = target_nr_racks
    nr_racks_per_pod = target_nr_racks_per_pod
    nr_pods = nr_racks // nr_racks_per_pod

    # Original data
    original_inner = data["matrix"]["inner"]
    
    # Validate original matrix dimensions
    if not original_inner or not isinstance(original_inner, list):
        raise ValueError(f"Invalid matrix format in {input_file}")
    
    # Validate matrix is square
    if len(original_inner) != len(original_inner[0]):
        raise ValueError(f"Matrix in {input_file} is not square: {len(original_inner)}x{len(original_inner[0])}")
    
    # Validate matrix dimensions match metadata
    original_nr_racks = data.get("nr_racks", len(original_inner))
    if original_nr_racks != len(original_inner):
        print(f"Warning: Matrix dimensions ({len(original_inner)}) don't match metadata ({original_nr_racks}) in {input_file}")
    
    # Scale the data to match target dimensions while preserving patterns
    scaled_inner = scale_inner(original_inner, nr_racks, nr_racks)

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

    # Create the scaled data
    scaled_data = {
        "matrix": {
            "inner": scaled_inner,
            "idx2name": idx2name,
        },
        "pod2tors": new_pod2tors,
        "nr_pods": nr_pods,
        "nr_racks": nr_racks,
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the scaled data to a new file
    with open(output_file, "w") as file:
        json.dump(scaled_data, file, indent=4)

    print(f"Scaled data saved to {output_file}")
    print(f"Original dimensions: {len(original_inner)}x{len(original_inner[0])}")
    print(f"Scaled dimensions: {len(scaled_inner)}x{len(scaled_inner[0])}")
    print(f"Target dimensions: {nr_racks} racks, {nr_pods} pods, {nr_racks_per_pod} racks per pod")


def main():
    # Base directory
    base_dir = "/data1/lichenni/m4/parsimon-eval/workload/spatials"
    
    # Get target shape from cluster_b.json
    target_file = os.path.join(base_dir, "cluster_b.json")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Target file not found: {target_file}")
        
    with open(target_file, "r") as file:
        target_data = json.load(file)
        target_nr_racks = target_data["nr_racks"]
        target_nr_pods = target_data["nr_pods"]
        target_nr_racks_per_pod = target_nr_racks // target_nr_pods
    
    # Validate target data
    if target_nr_racks % target_nr_pods != 0:
        raise ValueError(f"Invalid target data: nr_racks ({target_nr_racks}) must be divisible by nr_pods ({target_nr_pods})")
    
    print(f"Target dimensions: {target_nr_racks} racks, {target_nr_pods} pods, {target_nr_racks_per_pod} racks per pod")
    
    # Upsample cluster_a.json to match cluster_b.json
    input_file_a = os.path.join(base_dir, "cluster_a.json")
    output_file_a = os.path.join(base_dir, "cluster_a_upsample.json")
    if not os.path.exists(input_file_a):
        raise FileNotFoundError(f"Input file not found: {input_file_a}")
    upsample_cluster(input_file_a, output_file_a, target_nr_racks, target_nr_racks_per_pod)
    
    # Upsample cluster_c.json to match cluster_b.json
    input_file_c = os.path.join(base_dir, "cluster_c.json")
    output_file_c = os.path.join(base_dir, "cluster_c_upsample.json")
    if not os.path.exists(input_file_c):
        raise FileNotFoundError(f"Input file not found: {input_file_c}")
    upsample_cluster(input_file_c, output_file_c, target_nr_racks, target_nr_racks_per_pod)


if __name__ == "__main__":
    main()

import pandas as pd
from pymatgen.core import Structure, Element
from pymatgen.ext.matproj import MPRester
import warnings
import os
import json
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import KMeans


def load_material_ids(filepath):
    
    try:
        materials_df = pd.read_csv(filepath)
        material_ids = materials_df['mpid'].tolist()
        return material_ids
    except Exception as e:
        print(f"Error loading material ids from {filepath}: {e}")
        return []

def get_structure(material_id):
   
    try:
        structure = mpr.get_structure_by_material_id(material_id)
        return structure
    except Exception as e:
        print(f"Error fetching structure for material ID {material_id}: {e}")
        return None

def get_atomic_radius(z):
    """Get the atomic radius for an element using pymatgen."""
    try:
        element = Element.from_Z(z)
        radius = (element.atomic_radius_covalent or element.atomic_radius or
                  element.atomic_radius_calculated or 1.5)
        return radius
    except Exception:
        return 1.5

def crystal_graph_coordinator():
   
    A = structure.lattice.matrix
    Z = [site Z for site in structure]
    X = structure.frac_coords
    cart_coords = structure.cart_coords  

    V_c = np.linalg.det(A)
    r = [get_atomic_radius(z) for z in Z]
    V_a = sum((4 / 3) * np.pi * ri**3 for ri in r)
    f_v = (V_c / V_a)**(1 / 3)

    neighbor_lists = []
    edge_type_counts = []
    all_neighbors = []
    bond_lengths = []

    for i in range(len(Z)):
        J, D, B = [], [], []
        for j in range(len(Z)):
            c = (r[i] + r[j]) * f_v
            for n in [np.zeros(3)]:  
                d = np.linalg.norm(A (X[j] - X[i] + n))
                if d < c and i != j:
                    J.append(j)
                    D.append(d / c)  # Normalized distance
                    B.append(d)      # Actual bond length

        type_counts = [0, 0]
        sorted_neighbors = []
        sorted_bond_lengths = []

        if len(D) > 1:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(D).reshape(-1, 1))
            C1 = np.where(kmeans.labels_ == 0)[0]
            C2 = np.where(kmeans.labels_ == 1)[0]
            delta = np.mean(np.array(D)[C2]) - np.mean(np.array(D)[C1])

            if delta < beta:
                # Keep only C1 edges (type 0)
                sorted_neighbors = [J[k] for k in C1]
                sorted_bond_lengths = [B[k] for k in C1]
                type_counts[0] = len(C1)
            else:
                # Split into type 0 (C1) and type 1 (C2)
                sorted_neighbors = [J[k] for k in C1] + [J[k] for k in C2]
                sorted_bond_lengths = [B[k] for k in C1] + [B[k] for k in C2]
                type_counts[0] = len(C1)
                type_counts[1] = len(C2)
        else:
            # All edges are type 0
            sorted_neighbors = J
            sorted_bond_lengths = B
            type_counts[0] = len(J)

        neighbor_lists.append(sorted_neighbors)
        edge_type_counts.append(type_counts)
        all_neighbors.extend(sorted_neighbors)
        bond_lengths.extend(sorted_bond_lengths)

    # Compute global type counts and neighbor counts matrix
    global_type_counts = [
        sum(tc[0] for tc in edge_type_counts),
        sum(tc[1] for tc in edge_type_counts)
    ]
    neighbor_counts = np.array(edge_type_counts, dtype=np.int32).T  # Shape: (2, num_nodes)

    return {
        "neighbor_lists": neighbor_lists,
        "global_type_counts": global_type_counts,
        "neighbor_counts": neighbor_counts,
        "all_neighbors": np.array(all_neighbors, dtype=np.int32),
        "bond_lengths": np.array(bond_lengths, dtype=np.float32),
        "cart_coords": cart_coords.astype(np.float32)  
    }

def process_material(material_id, count, total_count):
    """Convert materials into graph data with edge types, bond lengths, and positions."""
    structure = get_structure(material_id)
    if structure is None:
        print(f"Skipping material ID {material_id} due to structure retrieval error.")
        return None

    node_features = [site.specie.Z for site in structure]
    cg_data = crystal_graph_coordinator(structure)

    # Extract graph components
    type_counts = cg_data["global_type_counts"]
    neighbor_counts = cg_data["neighbor_counts"]
    neighbors = cg_data["all_neighbors"]
    bond_lengths = cg_data["bond_lengths"]
    neighbor_lists = cg_data["neighbor_lists"]
    cart_coords = cg_data["cart_coords"]  
    # Compute max neighbors
    max_count = max(len(nbrs) for nbrs in neighbor_lists) if neighbor_lists else 0

    print(f"Processed {count} out of {total_count} materials: {material_id}")

    return {
        "mpid": material_id,
        "node_features": np.array(node_features, dtype=np.int32),
        "type_counts": np.array(type_counts, dtype=np.int32),
        "neighbor_counts": neighbor_counts,
        "neighbors": neighbors,
        "bond_lengths": bond_lengths,
        "cart_coords": cart_coords,  
        "max_neighbors": max_count
    }

def main(csv_file, output_file, num_cpus): 
  
    material_ids = load_material_ids(csv_file)
    if not material_ids:
        print("No valid material ids found.")
        

    with parallel_backend('threading', n_jobs=num_cpus):
        results = Parallel(n_jobs=num_cpus)(
            delayed(process_material)(mid, i + 1, len(material_ids))
            for i, mid in enumerate(material_ids)
        )

    results = [res for res in results if res is not None]
    processed_graphs = {}
    max_neighbors = 0
    unique_z = set()

    for res in results:
        material_id = res["mpid"]
        processed_graphs[material_id] = {
            "node_features": res["node_features"],
            "type_counts": res["type_counts"],
            "neighbor_counts": res["neighbor_counts"],
            "neighbors": res["neighbors"],
            "bond_lengths": res["bond_lengths"],
            "cart_coords": res["cart_coords"] 
        }
        unique_z.update(res["node_features"])
        max_neighbors = max(max_neighbors, res["max_neighbors"])

    if processed_graphs:
        np.savez_compressed(output_file, graph_dict=processed_graphs)
        print(f"Graph data saved to {output_file} with {len(processed_graphs)} graphs.")

    num_z = len(unique_z)
    config = {
        "atomic_numbers": [int(z) for z in sorted(unique_z)],
        "node_vectors": np.eye(num_z, num_z).tolist(),
        "max_neighbors": int(max_neighbors),
        "pos_dim": 3
    }
    config_file = os.path.splitext(output_file)[0] + "_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_file}")


if __name__ == "__main__":
    csv_path = "file.csv"
    output_npz = "file.npz"
    main(csv_path, output_npz, num_cpus=num_cpus)

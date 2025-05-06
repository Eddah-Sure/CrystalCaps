if __name__ == "__main__":
    dataset_path = ""

    model, results = run_spatial_gnn_capsnet(
        dataset_path,
        target_name="e_form",
        epochs=epochs      
    )

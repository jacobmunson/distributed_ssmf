import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def scale_trip_data(df: pd.DataFrame, scaling_factor: int, noise_level: float, drop_rate: float) -> pd.DataFrame:
    """
    Scales a taxi trip dataset by creating perturbed copies across both matrix dimensions.

    This function takes an original DataFrame and generates a new, larger DataFrame
    by creating 'scaling_factor' copies. 
    
    Each copy is modified by:
    1. Offsetting its row/col IDs ('PU_idx', 'DO_idx') to ensure uniqueness.
    2. Keeping the temporal ID ('t_idx') the same to increase matrix size.
    3. Perturbing the 'trip_count' values with multiplicative noise.
    4. Randomly dropping a fraction of rows to introduce variation.

    Args:
        df: The original DataFrame. 
            Must contain 'PU_idx', 'DO_idx', 't_idx', and 'trip_count'.
        scaling_factor: The number of times to replicate the data. 
            A factor of 10 will result in a DataFrame ~10x larger.
        noise_level: The standard deviation of the Gaussian noise 
            applied to trip counts (e.g., 0.1 for 10% noise).
        drop_rate: The fraction of rows to randomly drop 
            from each new copy (e.g., 0.05 for 5%).

    Returns:
        A new, scaled pandas DataFrame.
    """
    if scaling_factor <= 1:
        return df.copy()

    # Get the dimensions of the original "space" to create unique offsets
    d0_max = df['PU_idx'].max() + 1
    d1_max = df['DO_idx'].max() + 1
    t_max = df['t_idx'].max() + 1 
    print(f"Original dimensions: d0={d0_max}, d1={d1_max}, T={t_max}")
    
    # List to hold the original + all new scaled DataFrames
    all_dfs = [df.copy()]
    
    print(f"Generating {scaling_factor - 1} new copies of the data...")
    for i in tqdm(range(1, scaling_factor), desc="Scaling Data"):
        # Create a new copy for this iteration
        df_copy = df.copy()

        # 1. Create New Unique Spatial Identifiers 
        # Offset PU/DO IDs to create more rows/cols
        df_copy['PU_idx'] += i * d0_max
        df_copy['DO_idx'] += i * d1_max      

        #  2. Perturb the trip_count data 
        # Generate noise
        noise = np.random.normal(loc=0.0, scale=noise_level, size=len(df_copy))
        df_copy['trip_count'] = df_copy['trip_count'] * (1 + noise)
        
        # Ensure trip counts are at least 1 and are integers
        df_copy['trip_count'] = df_copy['trip_count'].clip(lower=1).round().astype(int)

        #  3. Insert randomness by dropping rows 
        if drop_rate > 0:
            df_copy = df_copy.sample(frac=1.0 - drop_rate)
            
        all_dfs.append(df_copy)

    print("Concatenating all dataframes...")
    # Combine all dataframes
    scaled_df = pd.concat(all_dfs, ignore_index=True)

    return scaled_df

def main():
    """Main function to parse arguments and run the scaling process."""
    parser = argparse.ArgumentParser(
        description="Scale a parquet trip dataset by increasing the density of each time slice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_file", 
        help="Path to the input parquet file."
    )
    parser.add_argument(
        "output_file", 
        help="Path to save the scaled output parquet file."
    )
    parser.add_argument(
        "--scaling-factor", 
        type=int, 
        default=10,
        help="How many times to replicate the data. A factor of 10 creates a ~10x larger dataset."
    )
    parser.add_argument(
        "--noise-level", 
        type=float, 
        default=0.15,
        help="Standard deviation of multiplicative noise to apply to trip counts (e.g., 0.15 for 15%% noise)."
    )
    parser.add_argument(
        "--drop-rate", 
        type=float, 
        default=0.1,
        help="Fraction of data to randomly drop from each new copy to introduce sparsity (e.g., 0.1 for 10%%)."
    )
    args = parser.parse_args()

    print(f"Loading data from '{args.input_file}'...")
    try:
        original_df = pd.read_parquet(args.input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Original dataset has {len(original_df):,} rows.")
    
    # Run scaling logic
    scaled_df = scale_trip_data(original_df, args.scaling_factor, args.noise_level, args.drop_rate)
    
    print(f"\nNew scaled dataset has {len(scaled_df):,} rows.")
    
    new_d0 = scaled_df['PU_idx'].max() + 1
    new_d1 = scaled_df['DO_idx'].max() + 1
    new_t = scaled_df['t_idx'].max() + 1
    
    print(f"New scaled dimensions: d0={new_d0}, d1={new_d1}, T={new_t}")
    print("(Note: Time dimension T is unchanged, as requested)")
    
    print(f"Saving scaled data to '{args.output_file}'...")
    try:
        scaled_df.to_parquet(args.output_file, index=False)
        print("Done!")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
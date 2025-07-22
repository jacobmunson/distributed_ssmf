# distributed_ssmf
This repository houses a Distributed Shifting Seasonal Matrix Factorization model.

## Under Construction - pending local HPC maintenance 

## To install:
```
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

## Activate environment & dependencies:
```
source ./activate_distributed_env.sh
```
(see the referenced file for specific module dependencies)

## Download & Clean Data:

Source data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

I used these files:

- yellow_tripdata_2020-03.parquet
- yellow_tripdata_2020-04.parquet
- green_tripdata_2020-03.parquet
- green_tripdata_2020-04.parquet
- fhv_tripdata_2020-03.parquet
- fhv_tripdata_2020-04.parquet

Clean data with notebook:
```
nytaxi_processing.ipynb
```

### Create Scaled Dataset (optional)

This is required for scaling experiments, but can be skipped when just getting the model up and running on the above dataset(s) produced from the notebook.

Use the script:
```
scale_data.py
```

Example run to create a new synthetic dataset with 10x matrix dimensions:
```
python scale_dataset.py taxi_yellow_green_rideshare_march_to_apr2020_triplets.parquet scaled_data10x.parquet --scaling-factor 10
```

## Models

### 1. Baseline SSMF model:

See:
  - Repo: https://github.com/kokikwbt/ssmf/tree/main
  - Paper: https://proceedings.neurips.cc/paper/2021/hash/1fb2a1c37b18aa4611c3949d6148d0f8-Abstract.html

Our slight alteration designed to accumulate forecasts at each step:
```
ssmf_forecast.py
```
This uses `ncp.py` from the original SSMF repository. We include a slightly (non-substantively) modified version in this repository.

### 2. SSMF Tuples

This is our version of SSMF that accepts a tuple stream of input. Can be found in:
```
ssmf_tuples.py
```
You can run via:
```
python ssmf_tuples.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet
```

### 3. Distributed SSMF

Our Distributed SSMF model is found in:
```
ssmf_mpi_2d_dist_init.py
```
This includes the Distributed NCP initialization model `ncp_distributed_2d()` , which is in the above script. Supporting functions are found in:
```
ncp_distributed.py
```

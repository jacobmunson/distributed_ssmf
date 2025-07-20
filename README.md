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

### Baseline SSMF model:

See:
  - Repo: https://github.com/kokikwbt/ssmf/tree/main
  - Paper: https://proceedings.neurips.cc/paper/2021/hash/1fb2a1c37b18aa4611c3949d6148d0f8-Abstract.html

Our slight alteration designed to accumulate forecasts at each step:
```
ssmf_forecast.py
```
This uses `ncp.py` from the original SSMF repository. We include a slightly (non-substantively) modified version in this repository.

### Coming soon: SSMF Tuples, Distributed SSMF, DistributedNCP

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

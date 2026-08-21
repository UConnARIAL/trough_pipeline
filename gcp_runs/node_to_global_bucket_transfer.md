# Aggregating TCN Outputs from Multiple GCP VMs using Cloud Storage

## Overview

TCN tiles are processed simultaneously on multiple GCP VMs located in different regions.

Each VM produces output using the same directory structure:

```text
output/
└── gpkgs/
    ├── <tile_id>/
    ├── <tile_id>/
    └── ...
```
Our already created aggregation bucket is:
```
gs://ca-tcn-global-aggregation
```
can create if required : 
```
gcloud storage buckets create gs://ca-tcn-global-aggregation \
    --location=us-central1 \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --soft-delete-duration=0
```

### 1. On each VM

Give user access to bucket
```
gcloud storage buckets add-iam-policy-binding \
    gs://ca-tcn-global-aggregation \
    --member="user:USER_EMAIL" \
    --role="roles/storage.objectUser"
```

### 2. Grant access to the VMs
```
gcloud auth login --no-launch-browser
```

Check access
```
gcloud auth list
```
### 3. Check to see if the VM can see the bucket

```
gcloud storage ls gs://ca-tcn-global-aggregation/
```

### 4. Copy TCN Results from a VM
go to the tcn pipeline and 
```
gcloud storage rsync \
    outputs/gpkgs/ \
    gs://ca-tcn-global-aggregation/gpkgs/ \
    --recursive
```
### 5. The Transfer Can Be Run Repeatedly

The gcloud storage rsync operation is designed to be safely rerun.
For this workflow it is effectively idempotent/restartable:


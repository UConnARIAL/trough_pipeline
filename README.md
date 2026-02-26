## Design Decisions
1. Build our own pipe, code in this repo. 
2. Keep the model inference and the gpkg generation seperate
3. Break the gpkg generation to multiple stages to fit into the HPC 48h time limit.

4. 


# trough_pipeline design options 
Attempts to build big data pipeline for troughs on NSF HPC's

Two options for patching and stiching:

1.use the old MAPLE pipeline with HDF5

2.use newer rasterio / torchgeo / raster vision frameworks to mange the patching / stiching 
![image](https://github.com/user-attachments/assets/9b334f60-ea0e-4fd4-b80a-157f0db6dde3)



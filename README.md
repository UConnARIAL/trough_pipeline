## Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
### Motivation
Rapid permafrost thaw is fundamentally reorganizing Arctic hydrology, transforming isolated ice-wedge troughs into interconnected "Tundra Capillary Networks" (TCNs) that accelerate drainage and alter carbon cycling. However, quantifying these networks remains challenging, as current methods lack the scalability and generalization required to map subtle, interconnected features across heterogeneous landscapes. Here, we present a framework combining deep learning-based Vision Transformers (ViTs) with graph theory to map and quantify TCN structure directly from sub-meter satellite imagery. 

### Overall objective for final delivery:
Builds per-tile GeoPackages (components/edges/nodes + per-tile global_stats + XML)
and a master GeoPackage (global_stats rows for all tiles, global_poly polygons + XML per tile),
plus a whole-mosaic aggregation table global_stats_summary (1 row) with properly weighted averages.

Decisions:
  • Orientation: keep ONLY step-weighted mean on [0,180) (drop edge-weighted).
  • global_poly attributes: store averages instead of raw counts for nodes/edges:
      - avg_graph_nodes_per_component = num_graph_nodes / num_components
      - avg_graph_edges_per_component = num_graph_edges / num_components

### Design Decisions to build pipeline
1. Build our own pipe, code in this repo. 
2. Keep the model inference and the gpkg generation seperate
3. Break the gpkg generation to multiple stages to fit into the HPC 48h time limit.
4. Stick to PGC the imagery mosaic structure (tile/subtile) for processing and also for final delivery 

### How to use multi step approach 
1. Inference via gpu
2. Graph processing via high mememory cpu nodes

config.toml maps the input/output for the multiple steps. i.e source/sink pattern

Create the two seperate conda env for the gpu / cpu processes using the env.yml

Run the inference pipline and then run each of the steps 0..3 generate gpkgs
0. filter the inferences 
1. subtile grpah theory
2. subtile summeries
3. global summary

### AK TCN data procesing aprox. costs*

|  | images | masks | gpkgs |
|-----:|------:|------:|------:|
| Total Size | 46T | 49G | 2.2T |
| Comp Cost in SU's | | ~3000 | ~48,000 |  
| TACC Frontera queue | | RTX (gpu) | NVDIMM (cpu) |  

*Above does not include the model training cost


### TCN design options that were considered / compared 
Attempts to build big data pipeline for troughs on NSF HPC's

Two options for patching and stiching:

1.use the old MAPLE pipeline with HDF5

2.use newer rasterio / torchgeo / raster vision frameworks to mange the patching / stiching 
![image](https://github.com/user-attachments/assets/9b334f60-ea0e-4fd4-b80a-157f0db6dde3)



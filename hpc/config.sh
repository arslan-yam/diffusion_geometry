# ---- edit these two lines once ----------------------------------------------
export ACCOUNT=proj_XXXX                       # your project ID, from `mp` on the cluster
export DG_REPO=$HOME/DiffusionGeometry         # where the authors' package is cloned
# ------------------------------------------------------------------------------
export ENV_NAME=dgeo                           # conda env name
export GPU_CONSTRAINT="type_a|type_b|type_c"   # V100 nodes
export CPU_CONSTRAINT="type_d"                 # 48-core, 768 GB nodes (for the original package)

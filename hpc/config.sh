# ---- edit these two lines once ----------------------------------------------
export ACCOUNT=${ACCOUNT:-$(cat ~/.dg_account 2>/dev/null || echo proj_XXXX)}   # your project ID: echo proj_NNNN > ~/.dg_account
export DG_REPO=$HOME/DiffusionGeometry         # where the authors' package is cloned
# ------------------------------------------------------------------------------
export ENV_NAME=dgeo                           # conda env name
export GPU_CONSTRAINT="type_a|type_b|type_c"   # V100 nodes
export CPU_CONSTRAINT="type_d"                 # 48-core, 768 GB nodes (for the original package)

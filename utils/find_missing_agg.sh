cd /scratch2/projects/PDG_shared/TCN_gpkgs

for d in */; do
    tile=${d%/}  # strip trailing slash
    if [ ! -f "$d/${tile}_TCN_aggregated.gpkg" ]; then
        echo "Missing aggregated GPKG in: $tile"
    fi
done
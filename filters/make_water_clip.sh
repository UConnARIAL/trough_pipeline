
SRC_GPKG=./water_3413.gpkg
SRC_LAYER=water_mask

AOI_GPKG=./study_area.gpkg
AOI_LAYER=study_area

ogr2ogr -f GPKG ./water_3413_clip2.gpkg "$SRC_GPKG" "$SRC_LAYER" \
  -clipsrc "$AOI_GPKG" -clipsrclayer "$AOI_LAYER" \
  -nln water_clip -overwrite

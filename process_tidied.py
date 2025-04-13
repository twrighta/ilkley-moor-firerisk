import rasterio as rio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.mask import mask
from rasterio.io import MemoryFile
import fiona
import numpy as np


# Function to read, clip, and reproject rasters
def process_rasters(vector_filepath, raster_filepath, dst_crs, output_filepath, r_output_file_ext):
    """
    :param vector_filepath: Vector file of shape to clip the raster into
    :param raster_filepath: Raster file you want to clip.
    :param dst_crs: Destination CRS string.
    :param output_filepath: Destination filepath (NO FILENAME AND EXTENSION)
    :param output_file_ext: String file name and extension - raster

    :return:
    """
    # Read clipping polygon
    with fiona.open(vector_filepath, "r") as vector_file:
        vector_polygon = [feature["geometry"] for feature in vector_file]  # Needs to be in list format for mask.mask

    # Read raster file
    with rio.open(raster_filepath) as raster_file:
        clipped_image, clip_transform = rio.mask.mask(raster_file, vector_polygon, crop=True, invert=False)
        clip_meta = raster_file.meta.copy()

        # Creates metadata of te clipped raster (Gtiff format)
        clip_meta.update({"driver": "GTiff",
                          "height": clipped_image.shape[1],
                          "width": clipped_image.shape[2],
                          "transform": clip_transform,
                          "nodata": -9999})
        # Set all unexpected/no data raster pixel values to 0. clipped_image is a numpy array so can use np operations
        clipped_image[np.isnan(clipped_image)] = 0
        clipped_image[clipped_image > 50] = 0
        clipped_image[clipped_image <= 0] = 0

        # Now load clipped raster as a memory file, to process further.
        with MemoryFile() as memfile:
            with memfile.open(**clip_meta) as clipped:
                clipped.write(clipped_image)

                # Reproject the clipped raster
                transform, width, height = calculate_default_transform(
                    clipped.crs, dst_crs, clipped.width, clipped.height, *clipped.bounds
                )

                reproj_meta = clipped.meta.copy()
                reproj_meta.update({
                    "driver": "GTiff",
                    "crs": dst_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "nodata": 0
                })

                # Write reprojected raster to disk
                with rio.open(output_filepath + r_output_file_ext, "w", **reproj_meta) as dst:
                    reproject(
                        source=rio.band(clipped, 1),
                        destination=rio.band(dst, 1),
                        src_transform=clipped.transform,
                        src_crs=clipped.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )


# Create rasters for Folium for the 4 time periods -------------------------------------------------------------------
STUDY_AREA_FILEPATH = "D:/Datascience/Datasets/Geospatial/Ilkley Moor Fire Risk/moorPolygonCollection.shp"
RASTER_OUT_FILEPATH = "D:/Datascience/QGIS Projects/Ilkley Fire Risk/Outputs/Reprojected Rasters for Folium/"

# 2020
process_rasters(STUDY_AREA_FILEPATH,
                "D:/Datascience/QGIS Projects/Ilkley Fire Risk/Outputs/2020_risk_scores_clipped.gpkg",
                "EPSG:3857",
                RASTER_OUT_FILEPATH,
                "clipped_2020_crs3857.tif")

# 2021
process_rasters(STUDY_AREA_FILEPATH,
                "D:/Datascience/QGIS Projects/Ilkley Fire Risk/Outputs/2021_risk_scores_clipped.gpkg",
                "EPSG:3857",
                RASTER_OUT_FILEPATH,
                "clipped_2021_crs3857.tif")

# 2022
process_rasters(STUDY_AREA_FILEPATH,
                "D:/Datascience/QGIS Projects/Ilkley Fire Risk/Outputs/2022_risk_scores_clipped.gpkg",
                "EPSG:3857",
                RASTER_OUT_FILEPATH,
                "clipped_2022_crs3857.tif")

# 2023
process_rasters(STUDY_AREA_FILEPATH,
                "D:/Datascience/QGIS Projects/Ilkley Fire Risk/Outputs/2023_risk_scores_clipped.gpkg",
                "EPSG:3857",
                RASTER_OUT_FILEPATH,
                "clipped_2023_crs3857.tif")

# All Years Averaged:
process_rasters(STUDY_AREA_FILEPATH,
                "D:/Datascience/QGIS Projects/Ilkley Fire Risk/Outputs/all_years_averaged_risk_scores_clipped.gpkg",
                "EPSG:3857",
                RASTER_OUT_FILEPATH,
                "all_years_averaged_risk_scores_clipped_crs3857.tif")

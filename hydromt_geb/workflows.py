import math

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer

def downscale(data, factor):
    return data.repeat(factor, axis=-2).repeat(factor, axis=-1)

def get_modflow_transform_and_shape(
    mask,
    hydro_crs,
    modflow_crs,
    modflow_resolution
) -> tuple[Affine, tuple[int, int]]:
    """
    Calculate modflow geotransformation and size of grid.

    Returns:
        modflow_affine: affine transformation for new MODFLOW grid.
        ncols: number of columns in the MODFLOW grid.
        nrows: number of rows in the MODFLOW grid.
    """

    hydro_transform = mask.raster.transform
    width = mask.raster.width
    height = mask.raster.height

    hydro_lon = [hydro_transform.c + hydro_transform.a * i for i in range(width)]
    hydro_lat = [hydro_transform.f + hydro_transform.e * i for i in range(height)]
    
    a, b = np.meshgrid(
        np.append(hydro_lon, 2*hydro_lon[-1] - hydro_lon[-2]),
        np.append(hydro_lat, 2*hydro_lat[-1] - hydro_lat[-2])
    )
    
    transformer = Transformer.from_crs(f"epsg:{hydro_crs}", f"epsg:{modflow_crs}")
    a_utm, b_utm = transformer.transform(b, a)

    xmin = np.min(a_utm)
    xmax = np.max(a_utm)
    ncols = math.ceil((xmax-xmin) / modflow_resolution)
    xmax = xmin + ncols * modflow_resolution
    ymin = np.min(b_utm)
    ymax = np.max(b_utm)
    nrows = math.ceil((ymax-ymin) / modflow_resolution)
    ymax = ymin + nrows * modflow_resolution

    gt_modlfow = (
        xmin, modflow_resolution, 0, ymax, 0, -modflow_resolution
    )

    modflow_affine = Affine.from_gdal(*gt_modlfow)

    return modflow_affine, (nrows, ncols)

def create_raster_shapefile(transform: Affine, shape: tuple[int, int], epsg: int) -> gpd.GeoDataFrame:
    """
    First creates a raster for the given transform and size, where the top-left cell is 0, and counting up, rows first. Then the raster is transformed into a GeoDataFrame (shapefile), where the numbers of the cells are used as IDs.

    Args:
        transform: geotransformation of the raster
        xsize: number of columns
        ysize: number of rows
        epsg: epsg of the raster

    Returns:
        gdf: GeoDataFrame with raster cells as geometries, each having a unique ID.
    """
    ysize, xsize = shape
    array = np.arange(ysize * xsize, dtype=np.int32).reshape((ysize, xsize))
    shapes = list(rasterio.features.shapes(array, transform=transform))
    geoms = [{'geometry': geom, 'properties': {'cell_id': int(v)}} for geom, v in shapes]
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=epsg)

    assert np.all(np.diff(gdf['cell_id']) >= 0) # check if cell_ids are sorted
    gdf = gdf.drop('cell_id', axis=1)

    gdf['x'] = np.tile(np.arange(xsize), ysize)
    gdf['y'] = np.arange(ysize).repeat(xsize)

    return gdf

def create_indices(hydro_transform, hydro_shape, hydro_epsg, modflow_transform, modflow_shape, modflow_epsg):
    """
    Creates a mapping of cells between MODFLOW and hydro, and saves them to npy-files. The mapping contains the x and y index of a hydro and MODFLOW cell and the area size (m\ :sup:`2`) of their overlap. Cell combinations that are not in the indices do not overlap.
    """

    # Create hydro shapefile
    hydro_gdf = create_raster_shapefile(hydro_transform, hydro_shape, hydro_epsg)
    hydro_gdf = hydro_gdf.to_crs(epsg=modflow_epsg)
    # Create MODFLOW shapefile
    modflow_gdf = create_raster_shapefile(modflow_transform, modflow_shape, modflow_epsg)

    hydro_gdf['hydro_geometry'] = hydro_gdf.geometry  # save geometry for after join
    # intersect hydro and MODFLOW shapefiles
    intersection = gpd.sjoin(modflow_gdf, hydro_gdf, how='inner', predicate='intersects', lsuffix='modflow', rsuffix='hydro')

    # calculate size of intersection
    intersection['area'] = intersection.apply(lambda x: x.hydro_geometry.intersection(x.geometry).area, axis=1)

    return intersection

def create_modflow_basin(hydro_mask, intersection, MODFLOW_shape) -> np.ndarray:
    """
    Creates a mask for the MODFLOW basin. All cells that have any area overlapping with any hydro cell are considered to be part of the MODFLOW basin.
    """
    # Creating 1D arrays containing ModFlow and hydro indices anf Interesected area [m2]
    ModFlow_index = np.array(intersection['y_modflow'] * MODFLOW_shape[1] + intersection['x_modflow'])
    hydro_index = np.array(intersection['y_hydro'] * hydro_mask.raster.width + intersection['x_hydro'])  # associated hydro cell index

    modflow_cell_area = np.bincount(
        ModFlow_index,
        weights=hydro_mask.data.ravel()[hydro_index] * intersection['area'],
        minlength=MODFLOW_shape[0] * MODFLOW_shape[1]
    ).reshape(MODFLOW_shape)
    return (modflow_cell_area > 0).astype(np.int8)
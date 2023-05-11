from typing import List, Optional
from hydromt.models.model_grid import GridMixin, GridModel
from hydromt.config import configread
import hydromt
import logging
import os
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine

from .workflows import downscale

logger = logging.getLogger(__name__)


class GEBModel(GridModel):
    _CLI_ARGS = {"region": "setup_grid", "res": "setup_basemaps"}
    
    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
        epsg=4326
    ):
        """Initialize a GridModel for distributed models with a regular grid."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

        self.epsg = epsg
        self.subgrid = GridMixin()

    def setup_grid(
        self,
        region: dict,
        sub_grid_factor: int,
        hydrography_fn: Optional[str] = None,
        basin_index_fn: Optional[str] = None,
    ) -> xr.DataArray:
        """Creates a 2D regular grid or reads an existing grid.
        An 2D regular grid will be created from a geometry (geom_fn) or bbox. If an existing
        grid is given, then no new grid will be generated.

        Adds/Updates model layers:
        * **grid** grid mask: add grid mask to grid object

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:
            * {'basin': [x, y]}

            Region must be of kind [grid, bbox, geom, basin, subbasin, interbasin].
        hydrography_fn : str
            Name of data source for hydrography data. Required if region is of kind 'basin', 'subbasin' or 'interbasin'.
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.

        Returns
        -------
        grid : xr.DataArray
            Generated grid mask.
        """
        self.logger.info(f"Preparing 2D grid.")
        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        if kind in ["basin", "subbasin"]:
            # retrieve global hydrography data (lazy!)
            ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
            if "bounds" not in region:
                region.update(basin_index=self.data_catalog[basin_index_fn])
            # get basin geometry
            geom, xy = hydromt.workflows.get_basin_geometry(
                ds=ds_org,
                kind=kind,
                logger=self.logger,
                **region
            )
            region.update(xy=xy)
            ds_hydro = ds_org.raster.clip_geom(geom)
        else:
            raise ValueError(
                f"Region for grid must of kind [basin, subbasin], kind {kind} not understood."
            )

        # Add region and grid to model
        self.set_geoms(geom, "region")
        
        ldd = ds_hydro['flwdir'].raster.reclassify(
            reclass_table=pd.DataFrame(
                index=[0, 1, 2, 4, 8, 16, 32, 64, 128],
                data={"ldd": [5, 6, 3, 2, 1, 4, 7, 8, 9]}
            ),
            method="exact"
        )['ldd']
        ldd.raster.set_nodata(0)
        
        self.set_grid(ldd, name='ldd')
        self.set_grid(ds_hydro['uparea'], name='upstream_area')
        self.set_grid(ds_hydro['elevtn'], name='elevation')
        self.set_grid(
            xr.where(ds_hydro['rivlen_ds'] != -9999, ds_hydro['rivlen_ds'], np.nan, keep_attrs=True),
            name='channel_length'
        )
        self.set_grid(ds_hydro['rivslp'], name='channel_slope')
        
        ds_hydro['mask'].raster.set_nodata(0)
        self.set_grid(ds_hydro['mask'].astype(np.int8), name='grid_mask')
        self.grid_coords = {d: self.grid['mask'].coords[d] for d in self.grid['mask'].dims}

        mask = self.grid['grid_mask']

        dst_transform = mask.raster.transform * Affine.scale(1 / sub_grid_factor)

        submask = hydromt.raster.full_from_transform(
            dst_transform,
            (mask.raster.shape[0] * sub_grid_factor, mask.raster.shape[1] * sub_grid_factor), 
            nodata=mask.raster.nodata,
            dtype=mask.dtype,
            name='grid_mask'
        )
        submask.data = downscale(mask.data, sub_grid_factor)

        self.subgrid.set_grid(submask)
        self.subgrid.grid_coords = {d: self.subgrid.grid['grid_mask'].coords[d] for d in self.subgrid.grid['grid_mask'].dims}
        self.subgrid.factor = sub_grid_factor

    def setup_cell_area_map(self):
        RADIUS_EARTH_EQUATOR = 40075017  # m
        distance_1_degree_latitude = RADIUS_EARTH_EQUATOR / 360

        mask = self.grid['grid_mask'].raster
        affine = mask.transform

        lat_idx = np.arange(0, mask.height).repeat(mask.width).reshape((mask.height, mask.width))
        lat = (lat_idx + 0.5) * affine.e + affine.f
        width_m = distance_1_degree_latitude * np.cos(np.radians(lat)) * abs(affine.a)
        height_m = distance_1_degree_latitude * abs(affine.e)

        cell_area = hydromt.raster.full(self.grid_coords, nodata=np.nan, dtype=np.float32, name='cell_area')
        cell_area.data = (width_m * height_m)
        self.set_grid(cell_area)

        sub_cell_area = hydromt.raster.full(
            self.subgrid.grid_coords,
            nodata=cell_area.raster.nodata,
            dtype=cell_area.dtype,
            name='cell_area'
        )

        sub_cell_area.data = downscale(cell_area.data, self.subgrid.factor)
        self.subgrid.set_grid(sub_cell_area)

    def setup_mannings(self):
        a = (2 * self.grid['cell_area']) / self.grid['upstream_area']
        a = xr.where(a > 1, 1, a)
        b = self.grid['elevation'] / 2000
        b = xr.where(b > 1, 1, b)
        
        mannings = hydromt.raster.full(self.grid_coords, nodata=np.nan, dtype=np.float32, name='mannings')
        mannings.data = 0.025 + 0.015 * a + 0.030 * b
        self.set_grid(mannings)

    def setup_channel_width(self, mimumum_width):
        channel_width_data = self.grid['upstream_area'] / 500
        channel_width_data = xr.where(channel_width_data < mimumum_width, mimumum_width, channel_width_data)
        
        channel_width = hydromt.raster.full(self.grid_coords, nodata=np.nan, dtype=np.float32, name='channel_width')
        channel_width.data = channel_width_data
        
        self.set_grid(channel_width)

    def setup_channel_depth(self):
        assert (self.grid['upstream_area'] > 0).all()
        channel_depth_data = 0.27 * self.grid['upstream_area'] ** 0.26
        channel_depth = hydromt.raster.full(self.grid_coords, nodata=np.nan, dtype=np.float32, name='channel_depth')
        channel_depth.data = channel_depth_data
        self.set_grid(channel_depth)

    def setup_channel_ratio(self):
        assert (self.grid['river_length'] > 0).all()
        channel_area = self.grid['channel_width'] * self.grid['channel_length']
        channel_ratio_data = channel_area / self.grid['cell_area']
        channel_ratio_data = xr.where(channel_ratio_data > 1, 1, channel_ratio_data)
        assert (channel_ratio_data >= 0).all()

        channel_ratio = hydromt.raster.full(self.grid_coords, nodata=np.nan, dtype=np.float32, name='channel_ratio')
        channel_ratio.data = channel_ratio_data
        self.set_grid(channel_ratio)

    def setup_elevation_STD(self):
        MERIT = self.data_catalog.get_rasterdataset("merit_hydro")
        # TODO: figure out half cell offset

        scaling = 10

        # find the upper left corner of the grid cells in self.grid
        y_step = self.grid.get_index('y')[1] - self.grid.get_index('y')[0]
        x_step = self.grid.get_index('x')[1] - self.grid.get_index('x')[0]
        upper_left_y = self.grid.get_index('y')[0] - y_step / 2
        upper_left_x = self.grid.get_index('x')[0] - x_step / 2
        ymin = np.isclose(MERIT.get_index('y'), upper_left_y, atol=abs(y_step) / 100).argmax()
        ymax = ymin + self.grid.mask.shape[0] * scaling
        xmin = np.isclose(MERIT.get_index('x'), upper_left_x, atol=abs(x_step) / 100).argmax()
        xmax = xmin + self.grid.mask.shape[1] * scaling

        # select data from MERIT using the grid coordinates
        high_res_elevation_data = MERIT.isel(
            y=slice(ymin, ymax),
            x=slice(xmin, xmax)
        )

        elevation_per_cell = (
            high_res_elevation_data.values.reshape(high_res_elevation_data.shape[0] // scaling, scaling, -1, scaling
        ).swapaxes(1, 2).reshape(-1, scaling, scaling))

        elevation_per_cell = high_res_elevation_data.values.reshape(high_res_elevation_data.shape[0] // scaling, scaling, -1, scaling).swapaxes(1, 2)

        standard_deviation = hydromt.raster.full(self.grid_coords, nodata=np.nan, dtype=np.float32, name='elevation_STD')
        standard_deviation.data = np.std(elevation_per_cell, axis=(2,3))
        self.set_grid(standard_deviation)

    def write_grid(
        self,
        variables=[],
        driver="GTiff",
        compress="deflate",
        **kwargs,
    ):
        self._assert_write_mode
        self.grid.raster.to_mapstack(self.root)
        self.subgrid.grid.raster.to_mapstack(self.root)

    def write(self):
        self.write_grid()


if __name__ == '__main__':
    yml = r"preprocessing/geb.yml"
    root = r"root"
    from preconfig import config, ORIGINAL_DATA, INPUT

    data_libs = [os.path.join(ORIGINAL_DATA, 'data_catalog.yml')]
    opt = configread(yml)
    
    geb_model = GEBModel(root=root, mode='w+', data_libs=data_libs)
    geb_model.build(opt=opt)
    # !hydromt build GEBModel root -d data.yml -i geb.yml

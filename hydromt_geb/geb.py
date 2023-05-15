from typing import List, Optional
from hydromt.models.model_grid import GridMixin, GridModel
import hydromt.workflows
import logging
import os
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
import matplotlib.pyplot as plt
import geopandas as gpd

from .workflows import downscale

logger = logging.getLogger(__name__)

class GEBModel(GridModel):
    _CLI_ARGS = {"region": "setup_grid"}
    
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
        self.MERIT_grid = GridMixin()
        self.table = {}

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
        
        self.set_grid(ldd, name='routing/kinematic/ldd')
        self.set_grid(ds_hydro['uparea'], name='routing/kinematic/upstream_area')
        self.set_grid(ds_hydro['elevtn'], name='landsurface/topo/elevation')
        self.set_grid(
            xr.where(ds_hydro['rivlen_ds'] != -9999, ds_hydro['rivlen_ds'], np.nan, keep_attrs=True),
            name='routing/kinematic/channel_length'
        )
        self.set_grid(ds_hydro['rivslp'], name='routing/kinematic/channel_slope')
        
        # ds_hydro['mask'].raster.set_nodata(-1)
        self.set_grid(ds_hydro['mask'].astype(np.int8), name='input/areamaps/grid_mask')

        mask = self.grid['input/areamaps/grid_mask']

        dst_transform = mask.raster.transform * Affine.scale(1 / sub_grid_factor)

        submask = hydromt.raster.full_from_transform(
            dst_transform,
            (mask.raster.shape[0] * sub_grid_factor, mask.raster.shape[1] * sub_grid_factor), 
            nodata=0,
            dtype=mask.dtype,
            name='input/areamaps/sub_grid_mask'
        )
        submask.raster.set_nodata(None)
        submask.data = downscale(mask.data, sub_grid_factor)

        self.subgrid.set_grid(submask)
        self.subgrid.factor = sub_grid_factor

    def setup_cell_area_map(self):
        RADIUS_EARTH_EQUATOR = 40075017  # m
        distance_1_degree_latitude = RADIUS_EARTH_EQUATOR / 360

        mask = self.grid['input/areamaps/grid_mask'].raster
        affine = mask.transform

        lat_idx = np.arange(0, mask.height).repeat(mask.width).reshape((mask.height, mask.width))
        lat = (lat_idx + 0.5) * affine.e + affine.f
        width_m = distance_1_degree_latitude * np.cos(np.radians(lat)) * abs(affine.a)
        height_m = distance_1_degree_latitude * abs(affine.e)

        cell_area = hydromt.raster.full(mask.coords, nodata=np.nan, dtype=np.float32, name='areamaps/cell_area')
        cell_area.data = (width_m * height_m)
        self.set_grid(cell_area)

        sub_cell_area = hydromt.raster.full(
            self.subgrid.grid.raster.coords,
            nodata=cell_area.raster.nodata,
            dtype=cell_area.dtype,
            name='areamaps/sub_cell_area'
        )

        sub_cell_area.data = downscale(cell_area.data, self.subgrid.factor)
        self.subgrid.set_grid(sub_cell_area)

    def setup_mannings(self):
        a = (2 * self.grid['areamaps/cell_area']) / self.grid['routing/kinematic/upstream_area']
        a = xr.where(a > 1, 1, a)
        b = self.grid['landsurface/topo/elevation'] / 2000
        b = xr.where(b > 1, 1, b)
        
        mannings = hydromt.raster.full(self.grid.raster.coords, nodata=np.nan, dtype=np.float32, name='routing/kinematic/mannings')
        mannings.data = 0.025 + 0.015 * a + 0.030 * b
        self.set_grid(mannings)

    def setup_channel_width(self, minimum_width):
        channel_width_data = self.grid['routing/kinematic/upstream_area'] / 500
        channel_width_data = xr.where(channel_width_data < minimum_width, minimum_width, channel_width_data)
        
        channel_width = hydromt.raster.full(self.grid.raster.coords, nodata=np.nan, dtype=np.float32, name='routing/kinematic/channel_width')
        channel_width.data = channel_width_data
        
        self.set_grid(channel_width)

    def setup_channel_depth(self):
        assert (self.grid['routing/kinematic/upstream_area'] > 0).all()
        channel_depth_data = 0.27 * self.grid['routing/kinematic/upstream_area'] ** 0.26
        channel_depth = hydromt.raster.full(self.grid.raster.coords, nodata=np.nan, dtype=np.float32, name='routing/kinematic/channel_depth')
        channel_depth.data = channel_depth_data
        self.set_grid(channel_depth)

    def setup_channel_ratio(self):
        assert (self.grid['routing/kinematic/channel_length'] > 0).all()
        channel_area = self.grid['routing/kinematic/channel_width'] * self.grid['routing/kinematic/channel_length']
        channel_ratio_data = channel_area / self.grid['areamaps/cell_area']
        channel_ratio_data = xr.where(channel_ratio_data > 1, 1, channel_ratio_data)
        assert (channel_ratio_data >= 0).all()

        channel_ratio = hydromt.raster.full(self.grid.raster.coords, nodata=np.nan, dtype=np.float32, name='routing/kinematic/channel_ratio')
        channel_ratio.data = channel_ratio_data
        self.set_grid(channel_ratio)

    def setup_elevation_STD(self):
        MERIT = self.data_catalog.get_rasterdataset("merit_hydro")
        # There is a half degree offset in MERIT data
        MERIT_x_step = MERIT.coords['x'][1] - MERIT.coords['x'][0]
        MERIT_y_step = MERIT.coords['y'][0] - MERIT.coords['y'][1]
        MERIT = MERIT.assign_coords(
            x=MERIT.coords['x'] + MERIT_x_step / 2,
            y=MERIT.coords['y'] + MERIT_y_step / 2
        )

        # we are going to match the upper left corners. So create a MERIT grid with the upper left corners as coordinates
        MERIT_ul = MERIT.assign_coords(
            x=MERIT.coords['x'] - MERIT_x_step / 2,
            y=MERIT.coords['y'] + MERIT_y_step / 2
        )

        scaling = 10

        # find the upper left corner of the grid cells in self.grid
        y_step = self.grid.get_index('y')[1] - self.grid.get_index('y')[0]
        x_step = self.grid.get_index('x')[1] - self.grid.get_index('x')[0]
        upper_left_y = self.grid.get_index('y')[0] - y_step / 2
        upper_left_x = self.grid.get_index('x')[0] - x_step / 2
        
        ymin = np.isclose(MERIT_ul.get_index('y'), upper_left_y, atol=MERIT_y_step.item() / 100)
        assert ymin.sum() == 1, "Could not find the upper left corner of the grid cell in MERIT data"
        ymin = ymin.argmax()
        ymax = ymin + self.grid.mask.shape[0] * scaling
        xmin = np.isclose(MERIT_ul.get_index('x'), upper_left_x, atol=MERIT_x_step.item() / 100)
        assert xmin.sum() == 1, "Could not find the upper left corner of the grid cell in MERIT data"
        xmin = xmin.argmax()
        xmax = xmin + self.grid.mask.shape[1] * scaling

        # select data from MERIT using the grid coordinates
        high_res_elevation_data = MERIT.isel(
            y=slice(ymin, ymax),
            x=slice(xmin, xmax)
        )

        self.MERIT_grid.set_grid(MERIT.isel(
            y=slice(ymin-1, ymax+1),
            x=slice(xmin-1, xmax+1)
        ), name='landsurface/topo/sub_elevation')

        elevation_per_cell = (
            high_res_elevation_data.values.reshape(high_res_elevation_data.shape[0] // scaling, scaling, -1, scaling
        ).swapaxes(1, 2).reshape(-1, scaling, scaling))

        elevation_per_cell = high_res_elevation_data.values.reshape(high_res_elevation_data.shape[0] // scaling, scaling, -1, scaling).swapaxes(1, 2)

        standard_deviation = hydromt.raster.full(self.grid.raster.coords, nodata=np.nan, dtype=np.float32, name='landsurface/topo/elevation_STD')
        standard_deviation.data = np.std(elevation_per_cell, axis=(2,3))
        self.set_grid(standard_deviation)

    def interpolate(self, ds, interpolation_method, ydim='y', xdim='x'):
        return ds.interp(
            method=interpolation_method,
            **{
                ydim: self.grid.coords['y'].values,
                xdim: self.grid.coords['x'].values
            }
        )

    def setup_soil_parameters(self, interpolation_method='nearest'):
        soil_ds = self.data_catalog.get_rasterdataset("cwatm_soil_5min")
        for parameter in ('alpha', 'ksat', 'lambda', 'thetar', 'thetas'):
            for soil_layer in range(1, 4):
                ds = soil_ds[f'{parameter}{soil_layer}_5min']
                self.set_grid(self.interpolate(ds, interpolation_method), name=f'soil/{parameter}{soil_layer}')

        for soil_layer in range(1, 3):
            ds = soil_ds[f'storageDepth{soil_layer}']
            self.set_grid(self.interpolate(ds, interpolation_method), name=f'soil/storage_depth{soil_layer}')

        ds = soil_ds['percolationImp']
        self.set_grid(self.interpolate(ds, interpolation_method), name=f'soil/percolation_impeded')
        ds = soil_ds['cropgrp']
        self.set_grid(self.interpolate(ds, interpolation_method), name=f'soil/cropgrp')

    def setup_land_use_parameters(self, interpolation_method='nearest'):
        for land_use_type, land_use_type_netcdf_name in (
            ('forest', 'Forest'),
            ('grassland', 'Grassland'),
            ('irrPaddy', 'irrPaddy'),
            ('irrNonPaddy', 'irrNonPaddy'),
        ):
            land_use_ds = self.data_catalog.get_rasterdataset(f"cwatm_{land_use_type}_5min")
            
            for parameter in ('maxRootDepth', 'rootFraction1'):
                self.set_grid(
                    self.interpolate(land_use_ds[parameter], interpolation_method),
                    name=f'landcover/{land_use_type}/{parameter}_{land_use_type}'
                )
            
            parameter = f'cropCoefficient{land_use_type_netcdf_name}_10days'               
            self.set_forcing(
                self.interpolate(land_use_ds[parameter], interpolation_method),
                name=f'landcover/{land_use_type}/{parameter}'
            )
            if land_use_type in ('forest', 'grassland'):
                parameter = f'interceptCap{land_use_type_netcdf_name}_10days'               
                self.set_forcing(
                    self.interpolate(land_use_ds[parameter], interpolation_method),
                    name=f'landcover/{land_use_type}/{parameter}'
                )

    def setup_albedo(self, interpolation_method='nearest'):
        albedo_ds = self.data_catalog.get_rasterdataset("cwatm_albedo_5min")
        self.set_forcing(
            self.interpolate(albedo_ds['albedoLand'], interpolation_method, ydim='lat', xdim='lon'),
            name='landsurface/albedo/albedo_land'
        )
        self.set_forcing(
            self.interpolate(albedo_ds['albedoWater'], interpolation_method, ydim='lat', xdim='lon'),
            name='landsurface/albedo/albedo_water'
        )

    def setup_waterbodies(self):
        # TODO: Check whether intersect is the right predicate
        print('todo ^')
        waterbodies = self.data_catalog.get_geodataframe(
            "hydro_lakes",
            geom=self.staticgeoms['region'],
            predicate="intersects",
            variables=['waterbody_id', 'waterbody_type', 'volume_total']
        ).set_index('waterbody_id')

        self.set_grid(self.grid.raster.rasterize(
            waterbodies,
            col_name='waterbody_id',
            nodata=-1,
            all_touched=True,
            dtype=np.int32
        ), name='routing/lakesreservoirs/lakesResID')
        self.subgrid.set_grid(self.subgrid.grid.raster.rasterize(
            waterbodies,
            col_name='waterbody_id',
            nodata=-1,
            all_touched=True,
            dtype=np.int32
        ), name='routing/lakesreservoirs/sublakesResID')

        command_areas = self.data_catalog.get_geodataframe("reservoir_command_areas", geom=self.region, predicate="intersects")
        command_areas = command_areas[~command_areas['waterbody_id'].isnull()].reset_index(drop=True)
        command_areas['waterbody_id'] = command_areas['waterbody_id'].astype(np.int32)
        command_areas['geometry_in_region_bounds'] = gpd.overlay(command_areas, self.region, how='intersection', keep_geom_type=False)['geometry']
        command_areas['area'] = command_areas.to_crs(3857).area
        command_areas['area_in_region_bounds'] = command_areas['geometry_in_region_bounds'].to_crs(3857).area
        areas_per_waterbody = command_areas.groupby('waterbody_id').agg({'area': 'sum', 'area_in_region_bounds': 'sum'})
        relative_area_in_region = areas_per_waterbody['area_in_region_bounds'] / areas_per_waterbody['area']

        self.set_grid(self.grid.raster.rasterize(
            command_areas,
            col_name='waterbody_id',
            nodata=-1,
            all_touched=True,
            dtype=np.int32
        ), name='routing/lakesreservoirs/command_areas')
        self.subgrid.set_grid(self.subgrid.grid.raster.rasterize(
            command_areas,
            col_name='waterbody_id',
            nodata=-1,
            all_touched=True,
            dtype=np.int32
        ), name='routing/lakesreservoirs/subcommand_areas')

        # set all lakes with command area to reservoir

        waterbodies['volume_flood'] = waterbodies['volume_total']
        waterbodies.loc[waterbodies.index.isin(command_areas['waterbody_id']), 'waterbody_type'] = 2
        custom_reservoir_capacity = self.data_catalog.get_geodataframe("custom_reservoir_capacity").set_index('waterbody_id')
        custom_reservoir_capacity = custom_reservoir_capacity[custom_reservoir_capacity.index != -1]

        # TODO: test this
        print('todo ^')
        waterbodies.update(custom_reservoir_capacity)
        waterbodies = waterbodies.drop('geometry', axis=1)

        self.set_table(waterbodies, name='routing/lakesreservoirs/basin_lakes_data')
        
    def write_grid(
        self,
        driver="GTiff",
        compress="deflate",
        **kwargs,
    ) -> None:
        self._assert_write_mode
        self.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
        if len(self.subgrid._grid) > 0:
            self.subgrid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
        if len(self.MERIT_grid._grid) > 0:
            self.MERIT_grid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)

    def write_forcing(self) -> None:
        self._assert_write_mode
        self.logger.info("Write forcing files")
        for var in self.forcing:
            forcing = self.forcing[var]
            path = os.path.join(self.root, var + '.nc')
            # get folder of path
            folder = os.path.dirname(path)
            os.makedirs(folder, exist_ok=True)
            forcing.to_netcdf(path, mode='w')

    def write_table(self):
        """Write model table data to csv file at <root>/<fn>

        key-word arguments are passed to :py:func:`pd.to_csv`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'table/table.csv'
        """
        if len(self.table) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.table.items():
                fn = os.path.join(name + '.csv')
                self.logger.debug(f"Writing file {fn}")
                data.to_csv(os.path.join(self.root, fn))

    def set_table(self, table, name):
        self.table[name] = table

    def write(self):
        self.write_forcing()
        self.write_grid()
        self.write_table()

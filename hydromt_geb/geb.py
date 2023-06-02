from typing import List, Optional
from hydromt.models.model_grid import GridMixin, GridModel
import hydromt.workflows
import logging
import os
import math
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
import geopandas as gpd
from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt

from .workflows import downscale, get_modflow_transform_and_shape, create_indices, create_modflow_basin, pad_xy, create_farms

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
        self.region_subgrid = GridMixin()
        self.MERIT_grid = GridMixin()
        self.MODFLOW_grid = GridMixin()
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
        assert sub_grid_factor > 10, "sub_grid_factor must be larger than 10, because this is the resolution of the MERIT high-res DEM"
        assert sub_grid_factor % 10 == 0, "sub_grid_factor must be a multiple of 10"
        self.subgrid_factor = sub_grid_factor

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
        self.set_geoms(geom, name="areamaps/region")
        
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
        self.set_grid(ds_hydro['mask'].astype(np.int8), name='areamaps/grid_mask')

        mask = self.grid['areamaps/grid_mask']

        dst_transform = mask.raster.transform * Affine.scale(1 / sub_grid_factor)

        submask = hydromt.raster.full_from_transform(
            dst_transform,
            (mask.raster.shape[0] * sub_grid_factor, mask.raster.shape[1] * sub_grid_factor), 
            nodata=0,
            dtype=mask.dtype,
            crs=mask.raster.crs,
            name='areamaps/sub_grid_mask'
        )
        submask.raster.set_nodata(None)
        submask.data = downscale(mask.data, sub_grid_factor)

        self.subgrid.set_grid(submask)
        self.subgrid.factor = sub_grid_factor

    def setup_cell_area_map(self):
        RADIUS_EARTH_EQUATOR = 40075017  # m
        distance_1_degree_latitude = RADIUS_EARTH_EQUATOR / 360

        mask = self.grid['areamaps/grid_mask'].raster
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

        sub_cell_area.data = downscale(cell_area.data, self.subgrid.factor) / self.subgrid.factor ** 2
        self.subgrid.set_grid(sub_cell_area)

    def setup_regions_and_land_use(self, level, river_threshold):
        regions = self.data_catalog.get_geodataframe(
            f"gadm_level{level}",
            geom=self.staticgeoms['areamaps/region'],
            predicate="intersects",
        )
        self.set_geoms(regions, name='areamaps/regions')

        land_use = self.data_catalog.get_rasterdataset(
            "esa_worldcover_2020_v100",
            geom=self.geoms['areamaps/regions'],
            buffer=200 # 2 km buffer
        )

        region_bounds = self.geoms['areamaps/regions'].total_bounds
        
        resolution_x, resolution_y = self.subgrid.grid['areamaps/sub_grid_mask'].rio.resolution()
        pad_minx = region_bounds[0] - abs(resolution_x) / 2.0
        pad_miny = region_bounds[1] - abs(resolution_y) / 2.0
        pad_maxx = region_bounds[2] + abs(resolution_x) / 2.0
        pad_maxy = region_bounds[3] + abs(resolution_y) / 2.0

        # TODO: Is there a better way to do this?
        padded_subgrid, self.region_subgrid.slice = pad_xy(
            self.subgrid.grid['areamaps/sub_grid_mask'].rio,
            pad_minx,
            pad_miny,
            pad_maxx,
            pad_maxy,
            return_slice=True
        )
        
        reprojected_land_use = land_use.raster.reproject_like(
            padded_subgrid,
            method='nearest'
        )

        region_raster = reprojected_land_use.raster.rasterize(
            self.geoms['areamaps/regions'],
            col_name='UID',
            all_touched=True,
        )
        self.region_subgrid.set_grid(region_raster, name='areamaps/region_subgrid')

        MERIT = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            variables=['upg'],
            bbox=padded_subgrid.rio.bounds(),
            buffer=300 # 3 km buffer
        )
        # There is a half degree offset in MERIT data
        MERIT = MERIT.assign_coords(
            x=MERIT.coords['x'] + MERIT.rio.resolution()[0] / 2,
            y=MERIT.coords['y'] - MERIT.rio.resolution()[1] / 2
        )

        # Assume all cells with at least x upstream cells are rivers.
        rivers = MERIT > river_threshold
        rivers = rivers.astype(np.int32)
        rivers = rivers.rio.set_nodata(-1)
        rivers = rivers.raster.reproject_like(reprojected_land_use, method='nearest')
        self.region_subgrid.set_grid(rivers, name='landcover/rivers')

        hydro_land_use = reprojected_land_use.raster.reclassify(
            pd.DataFrame.from_dict({
                    10: 0, # tree cover
                    20: 1, # shrubland
                    30: 1, # grassland
                    40: 1, # cropland, setting to non-irrigated. Initiated as irrigated based on agents
                    50: 4, # built-up 
                    60: 1, # bare / sparse vegetation
                    70: 1, # snow and ice
                    80: 5, # permanent water bodies
                    90: 1, # herbaceous wetland
                    95: 5, # mangroves
                    100: 1, # moss and lichen
                }, orient='index'
            ),
        )[0]  # TODO: check why dataset is returned instead of dataarray, also need again setting of no data and crs
        # set rivers to 5 (permanent water bodies)
        hydro_land_use = xr.where(rivers != 1, hydro_land_use, 5)
        hydro_land_use.rio.set_crs(reprojected_land_use.rio.crs)
        hydro_land_use.rio.set_nodata(-1)
        
        self.region_subgrid.set_grid(hydro_land_use, name='landsurface/full_region_land_use_classes')

        cultivated_land = xr.where((hydro_land_use == 1) & (reprojected_land_use == 40), 1, 0)
        cultivated_land = cultivated_land.rio.set_nodata(-1)
        cultivated_land.rio.set_crs(reprojected_land_use.rio.crs)
        cultivated_land.rio.set_nodata(-1)

        self.region_subgrid.set_grid(cultivated_land, name='landsurface/full_region_cultivated_land')

        hydro_land_use_region = hydro_land_use.isel(self.region_subgrid.slice)

        # TODO: Doesn't work when using the original array. Somehow the dtype is changed on adding it to the subgrid. This is a workaround.
        self.subgrid.set_grid(hydro_land_use_region.values, name='landsurface/land_use_classes')

        cultivated_land_region = cultivated_land.isel(self.region_subgrid.slice)

        # Same workaround as above
        self.subgrid.set_grid(cultivated_land_region.values, name='landsurface/cultivated_land')

    def clip_with_grid(self, ds, mask):
        cells_along_x = mask.sum(dim='x')
        minx = (cells_along_x > 0).argmax().item()
        maxx = cells_along_x.size - (cells_along_x[::-1] > 0).argmax().item()
        
        cells_along_y = mask.sum(dim='y')
        miny = (cells_along_y > 0).argmax().item()
        maxy = cells_along_y.size - (cells_along_y[::-1] > 0).argmax().item()

        bounds = {'x': slice(miny, maxy), 'y': slice(minx, maxx)}

        return ds.isel(bounds), bounds


    def setup_farmers(self):
        regions = self.geoms['areamaps/regions']
        regions_raster = self.region_subgrid.grid['areamaps/region_subgrid']
        farm_sizes_m2 = [50_000]
        cell_area = self.subgrid.grid['areamaps/sub_cell_area']
        
        farms = hydromt.raster.full_like(regions_raster, nodata=-1)
        
        all_agents = []
        total_agent_count = 0
        for region_id in regions['UID']:
            region = regions_raster == region_id
            region_clip, bounds = self.clip_with_grid(region, region)

            cultivated_land_region = self.region_subgrid.grid['landsurface/full_region_cultivated_land'].isel(bounds)
            cultivated_land_region = xr.where(region_clip, cultivated_land_region, 0)
            # TODO: Why does nodata value disappear?
           
            # This is a small simplification to take the average across the region. Of course not entirely true
            # but should be ok.
            average_cell_area = cell_area.where(region).mean().item()
            if np.isnan(average_cell_area):
                continue

            farm_sizes_n_cells = [math.floor(farm_size_m2 / average_cell_area) for farm_size_m2 in farm_sizes_m2]
            total_cultivated_land_cells = cultivated_land_region.where(region).sum()
            
            agent_count_region = math.floor(total_cultivated_land_cells / farm_sizes_n_cells[0])
            left_over_cells = int((total_cultivated_land_cells - agent_count_region * farm_sizes_n_cells[0]).compute().item())

            agents = pd.DataFrame(index=np.arange(agent_count_region), columns=['farm_size_n_cells'])
            all_agents.append(agents)

            agents['farm_size_n_cells'] = farm_sizes_n_cells[0]
            agents.loc[0, 'farm_size_n_cells'] = agents.loc[0, 'farm_size_n_cells'] + left_over_cells
            
            farms_region = create_farms(agents, cultivated_land_region)
            farms_region[farms_region != -1] += total_agent_count

            farms[bounds] = xr.where(region_clip, farms_region, farms.isel(bounds))

            total_agent_count += agent_count_region

        all_agents = pd.concat(all_agents, ignore_index=True)
        assert len(all_agents) == all_agents.index.max() + 1 == farms.max() + 1

        # TODO: Again why is dtype changed? And export doesn't work?
        farms = farms.isel(self.region_subgrid.slice)
        farms = xr.where(self.subgrid.grid['areamaps/sub_grid_mask'] == 1, farms.values, -1)

        self.subgrid.set_grid(farms.values, name='agents/farmers/farms')
        self.subgrid.grid['agents/farmers/farms'].rio.set_nodata(-1)

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
        MERIT = self.data_catalog.get_rasterdataset("merit_hydro", variables=['elv'])
        # There is a half degree offset in MERIT data
        MERIT = MERIT.assign_coords(
            x=MERIT.coords['x'] + MERIT.rio.resolution()[0] / 2,
            y=MERIT.coords['y'] - MERIT.rio.resolution()[1] / 2
        )

        # we are going to match the upper left corners. So create a MERIT grid with the upper left corners as coordinates
        MERIT_ul = MERIT.assign_coords(
            x=MERIT.coords['x'] - MERIT.rio.resolution()[0] / 2,
            y=MERIT.coords['y'] - MERIT.rio.resolution()[1] / 2
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
    
    def setup_modflow(self, epsg, resolution):
        modflow_affine, MODFLOW_shape = get_modflow_transform_and_shape(
            self.grid.mask,
            4326,
            epsg,
            resolution
        )
        modflow_mask = hydromt.raster.full_from_transform(
            modflow_affine,
            MODFLOW_shape,
            nodata=0,
            dtype=np.int8,
            name=f'groundwater/modflow/{resolution}m/modflow_mask',
            crs=epsg
        )

        intersection = create_indices(
            self.grid.mask.raster.transform,
            self.grid.mask.raster.shape,
            4326,
            modflow_affine,
            MODFLOW_shape,
            epsg
        )

        save_folder = Path(self.root, f'groundwater/modflow/{resolution}m')
        save_folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_folder, 'modflow_indices.npz'),
            y_modflow=intersection['y_modflow'],
            x_modflow=intersection['x_modflow'],
            y_hydro=intersection['y_hydro'],
            x_hydro=intersection['x_hydro'],
            area=intersection['area']
        )

        modflow_mask.data = create_modflow_basin(self.grid.mask, intersection, MODFLOW_shape)
        self.MODFLOW_grid.set_grid(modflow_mask, name=f'groundwater/modflow/{resolution}m/modflow_mask')

        MERIT = self.data_catalog.get_rasterdataset("merit_hydro", variables=['elv'])
        MERIT_x_step = MERIT.coords['x'][1] - MERIT.coords['x'][0]
        MERIT_y_step = MERIT.coords['y'][0] - MERIT.coords['y'][1]
        MERIT = MERIT.assign_coords(
            x=MERIT.coords['x'] + MERIT_x_step / 2,
            y=MERIT.coords['y'] + MERIT_y_step / 2
        )
        elevation_modflow = MERIT.raster.reproject_like(modflow_mask, method='average')

        self.MODFLOW_grid.set_grid(elevation_modflow, name=f'groundwater/modflow/{resolution}m/modflow_elevation')

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
        waterbodies = self.data_catalog.get_geodataframe(
            "hydro_lakes",
            geom=self.staticgeoms['areamaps/region'],
            predicate="intersects",
            variables=['waterbody_id', 'waterbody_type', 'volume_total', 'average_discharge', 'average_area']
        ).set_index('waterbody_id')

        self.set_grid(self.grid.raster.rasterize(
            waterbodies,
            col_name='waterbody_id',
            nodata=0,
            all_touched=True,
            dtype=np.int32
        ), name='routing/lakesreservoirs/lakesResID')
        self.subgrid.set_grid(self.subgrid.grid.raster.rasterize(
            waterbodies,
            col_name='waterbody_id',
            nodata=0,
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
        relative_area_in_region.name = 'relative_area_in_region'  # set name for merge

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
        waterbodies = waterbodies.merge(relative_area_in_region, left_index=True, right_index=True)

        custom_reservoir_capacity = self.data_catalog.get_geodataframe("custom_reservoir_capacity").set_index('waterbody_id')
        custom_reservoir_capacity = custom_reservoir_capacity[custom_reservoir_capacity.index != -1]

        waterbodies.update(custom_reservoir_capacity)
        waterbodies = waterbodies.drop('geometry', axis=1)

        self.set_table(waterbodies, name='routing/lakesreservoirs/basin_lakes_data')

    def setup_precip_forcing(
        self,
        starttime: str,
        endtime: str,
        precip_fn: str = "cmip6",
        precip_clim_fn: Optional[str] = None,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Setup gridded precipitation forcing at model resolution.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        precip_fn : str, default era5
            Precipitation data source, see data/forcing_sources.yml.

            * Required variable: ['precip']
        precip_clim_fn : str, default None
            High resolution climatology precipitation data source to correct precipitation,
            see data/forcing_sources.yml.

            * Required variable: ['precip']
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        if precip_fn is None:
            return
        mask = self.grid['areamaps/grid_mask']

        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=form
        # https://www.isimip.org/documents/413/ISIMIP3b_bias_adjustment_fact_sheet_Gnsz7CO.pdf

        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_tuple=(starttime, endtime),
            variables=["precip"],
        )

        if chunksize is not None:
            precip = precip.chunk({"time": chunksize})

        clim = None
        if precip_clim_fn != None:
            clim = self.data_catalog.get_rasterdataset(
                precip_clim_fn,
                geom=precip.raster.box,
                buffer=2,
                variables=["precip"],
            )

        precip_out = hydromt.workflows.forcing.precip(
            precip=precip,
            da_like=self.staticmaps[self._MAPS["elevtn"]],
            clim=clim,
            freq=timedelta(days=1),
            resample_kwargs=dict(label="right", closed="right"),
            logger=self.logger,
            **kwargs,
        )

        # Update meta attributes (used for default output filename later)
        precip_out.attrs.update({"precip_fn": precip_fn})
        if precip_clim_fn is not None:
            precip_out.attrs.update({"precip_clim_fn": precip_clim_fn})
        self.set_forcing(precip_out.where(mask), name="precip")
        
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
        if len(self.region_subgrid._grid) > 0:
            self.region_subgrid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
        if len(self.MERIT_grid._grid) > 0:
            self.MERIT_grid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
        if len(self.MODFLOW_grid._grid) > 0:
            self.MODFLOW_grid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)

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
        self.write_geoms(fn="{name}.geojson")
        self.write_forcing()
        self.write_grid()
        self.write_table()

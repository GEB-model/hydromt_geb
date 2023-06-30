from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from hydromt.models.model_grid import GridMixin, GridModel
import hydromt.workflows
import logging
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from urllib.parse import urlparse

# temporary fix for ESMF on Windows
os.environ['ESMFMKFILE'] = str(Path(os.__file__).parent.parent / 'Library' / 'lib' / 'esmf.mk')

import xesmf as xe
from affine import Affine
import geopandas as gpd
from datetime import timedelta, datetime
from calendar import monthrange
import matplotlib.pyplot as plt
from isimip_client.client import ISIMIPClient

from .workflows import repeat_grid, get_modflow_transform_and_shape, create_indices, create_modflow_basin, pad_xy, create_farms, calculate_cell_area

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
        # TODO: How to do this properly?
        self.subgrid._read = True
        self.subgrid.logger = self.logger
        self.region_subgrid = GridMixin()
        self.region_subgrid._read = True
        self.region_subgrid.logger = self.logger
        self.MERIT_grid = GridMixin()
        self.MERIT_grid._read = True
        self.MERIT_grid.logger = self.logger
        self.MODFLOW_grid = GridMixin()
        self.MODFLOW_grid._read = True
        self.MODFLOW_grid.logger = self.logger
        self.table = {}
        self.binary = {}
        self.dict = {}

        self.model_structure = {
            "geoms": {},
            "grid": {},
            "subgrid": {},
            "region_subgrid": {},
            "MERIT_grid": {},
            "MODFLOW_grid": {},
            "table": {},
            "binary": {},
            "dict": {},
            "forcing": {}
        }

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
        self.set_grid((~ds_hydro['mask']).astype(np.int8), name='areamaps/grid_mask')

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
        submask.data = repeat_grid(mask.data, sub_grid_factor)

        self.subgrid.set_grid(submask)
        self.subgrid.factor = sub_grid_factor

    def setup_crops(
            self,
            crop_ids,
            crop_variables,
            crop_prices=None,
            cultivation_costs=None,
        ):
        self.set_dict(crop_ids, name='crops/crop_ids')
        self.set_dict(crop_variables, name='crops/crop_variables')
        if crop_prices is not None:
            if isinstance(crop_prices, str):
                with open(Path(self.root, crop_prices), 'r') as f:
                    crop_prices_data = json.load(f)
                crop_prices = {
                    'time': crop_prices_data['time'],
                    'crops': {
                        crop_id: crop_prices_data['crops'][crop_name]
                        for crop_id, crop_name in crop_ids.items()
                    }
                }
            self.set_dict(crop_prices, name='crops/crop_prices')
        if cultivation_costs is not None:
            if isinstance(cultivation_costs, str):
                with open(Path(self.root, cultivation_costs)) as f:
                    cultivation_costs = json.load(f)
                cultivation_costs = {
                    'time': cultivation_costs['time'],
                    'crops': {
                        crop_id: cultivation_costs['crops'][crop_name]
                        for crop_id, crop_name in crop_ids.items()
                    }
                }
            self.set_dict(cultivation_costs, name='crops/cultivation_costs')

    def setup_cell_area_map(self):
        mask = self.grid['areamaps/grid_mask'].raster
        affine = mask.transform

        cell_area = hydromt.raster.full(mask.coords, nodata=np.nan, dtype=np.float32, name='areamaps/cell_area')
        cell_area.data = calculate_cell_area(affine, mask.shape)
        self.set_grid(cell_area)

        sub_cell_area = hydromt.raster.full(
            self.subgrid.grid.raster.coords,
            nodata=cell_area.raster.nodata,
            dtype=cell_area.dtype,
            name='areamaps/sub_cell_area'
        )

        sub_cell_area.data = repeat_grid(cell_area.data, self.subgrid.factor) / self.subgrid.factor ** 2
        self.subgrid.set_grid(sub_cell_area)

    def setup_regions_and_land_use(self, region_database='gadm_level1', unique_region_id='UID', river_threshold=100):
        regions = self.data_catalog.get_geodataframe(
            region_database,
            geom=self.staticgeoms['areamaps/region'],
            predicate="intersects",
        ).rename(columns={unique_region_id: 'region_id'})
        assert np.issubdtype(regions['region_id'].dtype, np.integer), "Region ID must be integer"
        assert 'ISO3' in regions.columns, f"Region database must contain ISO3 column ({self.data_catalog[region_database].path})"
        self.set_geoms(regions, name='areamaps/regions')

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
            return_slice=True,
            constant_values=1,
        )
        padded_subgrid.raster.set_nodata(-1)
        self.region_subgrid.set_grid(padded_subgrid, name='areamaps/region_mask')
        
        land_use = self.data_catalog.get_rasterdataset(
            "esa_worldcover_2020_v100",
            geom=self.geoms['areamaps/regions'],
            buffer=200 # 2 km buffer
        )
        reprojected_land_use = land_use.raster.reproject_like(
            padded_subgrid,
            method='nearest'
        )

        region_raster = reprojected_land_use.raster.rasterize(
            self.geoms['areamaps/regions'],
            col_name='region_id',
            all_touched=True,
        )
        self.region_subgrid.set_grid(region_raster, name='areamaps/region_subgrid')

        self.grid['areamaps/cell_area']
        padded_cell_area = self.grid['areamaps/cell_area'].rio.pad_box(*region_bounds)

        region_cell_area = calculate_cell_area(padded_cell_area.raster.transform, padded_cell_area.shape)

        region_cell_area_subgrid = hydromt.raster.full_from_transform(
            padded_cell_area.raster.transform * Affine.scale(1 / self.subgrid.factor),
            (padded_cell_area.raster.shape[0] * self.subgrid.factor, padded_cell_area.raster.shape[1] * self.subgrid.factor), 
            nodata=np.nan,
            dtype=padded_cell_area.dtype,
            crs=padded_cell_area.raster.crs,
            name='areamaps/sub_grid_mask'
        )

        region_cell_area_subgrid.data = repeat_grid(region_cell_area, self.subgrid.factor) / self.subgrid.factor ** 2
        region_cell_area_subgrid_clipped_to_region = region_cell_area_subgrid.raster.clip_bbox((pad_minx, pad_miny, pad_maxx, pad_maxy))
        
        # TODO: Why is everything set to nan if not using values?
        self.region_subgrid.set_grid(region_cell_area_subgrid_clipped_to_region.values, name='areamaps/region_cell_area_subgrid')

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

    def setup_economic_data(self):
        print('Setting up economic data')
        lending_rates = self.data_catalog.get_geodataframe('wb_lending_rate')
        inflation_rates = self.data_catalog.get_geodataframe('wb_inflation_rate')

        lending_rates_dict, inflation_rates_dict = {'data': {}}, { 'data': {}}
        years_lending_rates = [c for c in lending_rates.columns if c.isnumeric() and len(c) == 4 and int(c) >= 1900 and int(c) <= 3000]
        lending_rates_dict['time'] = years_lending_rates
        years_inflation_rates = [c for c in inflation_rates.columns if c.isnumeric() and len(c) == 4 and int(c) >= 1900 and int(c) <= 3000]
        inflation_rates_dict['time'] = years_inflation_rates
        for _, region in self.geoms['areamaps/regions'].iterrows():
            region_id = region['region_id']
            ISO3 = region['ISO3']

            lending_rates_country = (lending_rates.loc[lending_rates["Country Code"] == ISO3, years_lending_rates] / 100 + 1)  # percentage to rate
            assert len(lending_rates_country) == 1, f"Expected one row for {ISO3}, got {len(lending_rates_country)}"
            lending_rates_dict['data'][region_id] = lending_rates_country.iloc[0].tolist()

            inflation_rates_country = (inflation_rates.loc[inflation_rates["Country Code"] == ISO3, years_inflation_rates] / 100 + 1) # percentage to rate
            assert len(inflation_rates_country) == 1, f"Expected one row for {ISO3}, got {len(inflation_rates_country)}"
            inflation_rates_dict['data'][region_id] = inflation_rates_country.iloc[0].tolist()

        self.set_dict(inflation_rates_dict, name='economics/inflation_rates')
        self.set_dict(lending_rates_dict, name='economics/lending_rates')

    def setup_well_prices_by_reference_year(self, well_price, upkeep_price_per_m2, reference_year, start_year, end_year):
        # create dictory with prices for well_prices per year by applying inflation rates
        inflation_rates = self.dict['economics/inflation_rates']
        regions = list(inflation_rates['data'].keys())

        well_prices_dict = {
            'time': list(range(start_year, end_year + 1)),
            'data': {}
        }
        for region in regions:
            well_prices = pd.Series(index=range(start_year, end_year + 1))
            well_prices.loc[reference_year] = well_price
            
            for year in range(reference_year + 1, end_year + 1):
                well_prices.loc[year] = well_prices[year-1] * inflation_rates['data'][region][inflation_rates['time'].index(str(year))]
            for year in range(reference_year -1, start_year -1, -1):
                well_prices.loc[year] = well_prices[year+1] / inflation_rates['data'][region][inflation_rates['time'].index(str(year+1))]

            well_prices_dict['data'][region] = well_prices.tolist()

        self.set_dict(well_prices_dict, name='economics/well_prices')
            
        upkeep_prices_dict = {
            'time': list(range(start_year, end_year + 1)),
            'data': {}
        }
        for region in regions:
            upkeep_prices = pd.Series(index=range(start_year, end_year + 1))
            upkeep_prices.loc[reference_year] = upkeep_price_per_m2
            
            for year in range(reference_year + 1, end_year + 1):
                upkeep_prices.loc[year] = upkeep_prices[year-1] * inflation_rates['data'][region][inflation_rates['time'].index(str(year))]
            for year in range(reference_year -1, start_year -1, -1):
                upkeep_prices.loc[year] = upkeep_prices[year+1] / inflation_rates['data'][region][inflation_rates['time'].index(str(year+1))]

            upkeep_prices_dict['data'][region] = upkeep_prices.tolist()

        self.set_dict(upkeep_prices_dict, name='economics/upkeep_prices_well_per_m2')
         
    def clip_with_grid(self, ds, mask):
        assert ds.shape == mask.shape
        cells_along_y = mask.sum(dim='x').values.ravel()
        miny = (cells_along_y > 0).argmax().item()
        maxy = cells_along_y.size - (cells_along_y[::-1] > 0).argmax().item()
        
        cells_along_x = mask.sum(dim='y').values.ravel()
        minx = (cells_along_x > 0).argmax().item()
        maxx = cells_along_x.size - (cells_along_x[::-1] > 0).argmax().item()

        bounds = {'y': slice(miny, maxy), 'x': slice(minx, maxx)}

        return ds.isel(bounds), bounds

    def setup_farmers(self, irrigation_sources=None, n_seasons=1):
        regions = self.geoms['areamaps/regions']
        regions_raster = self.region_subgrid.grid['areamaps/region_subgrid']
        
        farms = hydromt.raster.full_like(regions_raster, nodata=-1)
        farmers = pd.read_csv(Path(self.root, '..', 'preprocessing', 'agents', 'farmers', 'farmers.csv'), index_col=0)
        
        for region_id in regions['region_id']:
            print(f"Creating farms for region {region_id}")
            region = regions_raster == region_id
            region_clip, bounds = self.clip_with_grid(region, region)

            cultivated_land_region = self.region_subgrid.grid['landsurface/full_region_cultivated_land'].isel(bounds)
            cultivated_land_region = xr.where(region_clip, cultivated_land_region, 0)
            # TODO: Why does nodata value disappear?                  
            farmers_region = farmers[farmers['region_id'] == region_id]
            farms_region = create_farms(farmers_region, cultivated_land_region, farm_size_key='area_n_cells')

            farms[bounds] = xr.where(region_clip, farms_region, farms.isel(bounds))
        
        farmers = farmers.drop('area_n_cells', axis=1)

        # TODO: Again why is dtype changed? And export doesn't work?
        farms_copy = farms.copy()
        farms_copy = xr.where(self.region_subgrid.grid['areamaps/region_mask'], -1, farms_copy)
        cut_farms = np.unique(farms_copy.values)
        cut_farms = cut_farms[cut_farms != -1]

        subgrid_farms = self.clip_with_grid(farms, self.region_subgrid.grid['areamaps/region_mask'])[0]

        subgrid_farms_in_study_area = xr.where(np.isin(subgrid_farms, cut_farms), -1, subgrid_farms)
        farmers = farmers[~farmers.index.isin(cut_farms)]

        remap_farmer_ids = np.full(farmers.index.max() + 2, -1, dtype=np.int32) # +1 because 0 is also a farm, +1 because no farm is -1, set to -1 in next step
        remap_farmer_ids[farmers.index] = np.arange(len(farmers))
        subgrid_farms_in_study_area = remap_farmer_ids[subgrid_farms_in_study_area.values]

        farmers = farmers.reset_index(drop=True)
        
        assert np.setdiff1d(np.unique(subgrid_farms_in_study_area), -1).size == len(farmers)
        assert farmers.iloc[-1].name == subgrid_farms_in_study_area.max()

        self.subgrid.set_grid(subgrid_farms_in_study_area.squeeze(), name='agents/farmers/farms')
        self.subgrid.grid['agents/farmers/farms'].rio.set_nodata(-1)

        crop_name_to_id = {
            crop_name: int(ID)
            for ID, crop_name in self.dict['crops/crop_ids'].items()
        }
        crop_name_to_id[np.nan] = -1
        for season in range(1, n_seasons + 1):
            farmers[f'season_#{season}_crop'] = farmers[f'season_#{season}_crop'].map(crop_name_to_id)

        if irrigation_sources:
            self.set_dict(irrigation_sources, name='agents/farmers/irrigation_sources')
            farmers['irrigation_source'] = farmers['irrigation_source'].map(irrigation_sources)

        for column in farmers.columns:
            self.set_binary(farmers[column], name=f'agents/farmers/{column}')


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
        
        ymin = np.isclose(MERIT_ul.get_index('y'), upper_left_y, atol=MERIT.rio.resolution()[1] / 100)
        assert ymin.sum() == 1, "Could not find the upper left corner of the grid cell in MERIT data"
        ymin = ymin.argmax()
        ymax = ymin + self.grid.mask.shape[0] * scaling
        xmin = np.isclose(MERIT_ul.get_index('x'), upper_left_x, atol=MERIT.rio.resolution()[0] / 100)
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
        ), name='landsurface/topo/subgrid_elevation')

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
            name=f'groundwater/modflow/modflow_mask',
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

        self.set_binary(intersection['y_modflow'], name=f'groundwater/modflow/y_modflow')
        self.set_binary(intersection['x_modflow'], name=f'groundwater/modflow/x_modflow')
        self.set_binary(intersection['y_hydro'], name=f'groundwater/modflow/y_hydro')
        self.set_binary(intersection['x_hydro'], name=f'groundwater/modflow/x_hydro')
        self.set_binary(intersection['area'], name=f'groundwater/modflow/area')

        modflow_mask.data = create_modflow_basin(self.grid.mask, intersection, MODFLOW_shape)
        self.MODFLOW_grid.set_grid(modflow_mask, name=f'groundwater/modflow/modflow_mask')

        MERIT = self.data_catalog.get_rasterdataset("merit_hydro", variables=['elv'])
        MERIT_x_step = MERIT.coords['x'][1] - MERIT.coords['x'][0]
        MERIT_y_step = MERIT.coords['y'][0] - MERIT.coords['y'][1]
        MERIT = MERIT.assign_coords(
            x=MERIT.coords['x'] + MERIT_x_step / 2,
            y=MERIT.coords['y'] + MERIT_y_step / 2
        )
        elevation_modflow = MERIT.raster.reproject_like(modflow_mask, method='average')

        self.MODFLOW_grid.set_grid(elevation_modflow, name=f'groundwater/modflow/modflow_elevation')

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

    def download_isimip(self, starttime, endtime, variable, forcing, resolution=None, buffer=0):
        client = ISIMIPClient()
        download_path = Path(self.root).parent / 'preprocessing' / 'climate' / forcing / variable
        download_path.mkdir(parents=True, exist_ok=True)
        # get the dataset metadata from the ISIMIP repository
        response = client.datasets(
            simulation_round='ISIMIP3a',
            product='InputData',
            climate_forcing=forcing,
            climate_scenario='obsclim',
            climate_variable=variable,
            resolution=resolution,
        )
        assert len(response["results"]) == 1
        dataset = response["results"][0]
        files = dataset['files']

        xmin, ymin, xmax, ymax = self.bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer
        
        download_files = []
        parse_files = []
        for file in files:
            name = file['name']
            assert name.endswith('.nc')
            splitted_filename = name.split('_')
            date = splitted_filename[-1].split('.')[0]
            if len(date) == 6:
                start_year = int(date[:4])
                end_year = start_year
                month = int(date[4:6])
            elif len(date) == 4:
                start_year = int((splitted_filename[-2]))
                end_year = int(date[:4])
                month = None
            else:
                raise ValueError(f'could not parse date {date} from file {name}')

            if (not (endtime.year < start_year or starttime.year > end_year)) and (month is None or (month >= starttime.month and month <= endtime.month)):
                parse_files.append(file['name'].replace('_global', f'_lat{ymin}to{ymax}lon{xmin}to{xmax}'))
                if not (download_path / name.replace('_global', f'_lat{ymin}to{ymax}lon{xmin}to{xmax}')).exists():
                    download_files.append(file['path'])

        if download_files:
            response = client.cutout(download_files, [ymin, ymax, xmin, xmax], poll=10)
            # download the file when it is ready
            client.download(
                response['file_url'],
                path=download_path,
                validate=False,
                extract=True
            )
            # remove zip file
            (download_path / Path(urlparse(response['file_url']).path.split('/')[-1])).unlink()
            
        datasets = [xr.open_dataset(download_path / file) for file in parse_files]
        coords_first_dataset = datasets[0].coords
        for dataset in datasets:
            # make sure all datasets have more or less the same coordinates
            assert np.isclose(dataset.coords['lat'].values, coords_first_dataset['lat'].values, atol=abs(datasets[0].rio.resolution()[1] / 100), rtol=0).all()
            assert np.isclose(dataset.coords['lon'].values, coords_first_dataset['lon'].values, atol=abs(datasets[0].rio.resolution()[0] / 100), rtol=0).all()

        datasets = [
            ds.assign_coords(
                lon=coords_first_dataset['lon'],
                lat=coords_first_dataset['lat'],
                inplace=True
            ) for ds in datasets
        ]
        ds = xr.concat(datasets, dim='time').sel(time=slice(starttime, endtime))

        # assert that time is monotonically increasing with a constant step size
        assert (ds.time.diff('time').astype(np.int64) == (ds.time[1] - ds.time[0]).astype(np.int64)).all()
        ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        return ds

    def setup_forcing(
            self,
            starttime,
            endtime,
        ):
        # download source data from ISIMIP
        high_res_variables = ['pr', 'rsds', 'tas', 'tasmax', 'tasmin']
        low_res_variables = ['hurs', 'sfcwind', 'rlds', 'ps']

        self.setup_high_resolution_variables(high_res_variables, starttime, endtime)
        self.setup_hurs(starttime, endtime)
        self.setup_longwave(starttime=starttime, endtime=endtime)
        self.setup_pressure(starttime, endtime)
        self.setup_wind(starttime, endtime)

    def setup_longwave(self, starttime, endtime):
        x1 = 0.43
        x2 = 5.7
        sbc = 5.67E-8   # stefan boltzman constant [Js−1 m−2 K−4]

        es0 = 6.11  # reference saturation vapour pressure  [hPa]
        T0 = 273.15
        lv = 2.5E6  # latent heat of vaporization of water
        Rv = 461.5  # gas constant for water vapour [J K kg-1]

        target = self.grid['landsurface/topo/elevation'].rename({'x': 'lon', 'y': 'lat'})

        hurs_coarse = self.download_isimip(variable='hurs', starttime=starttime, endtime=endtime, forcing='gswp3-w5e5', buffer=1)  # some buffer to avoid edge effects / errors in ISIMIP API
        tas_coarse = self.download_isimip(variable='tas', starttime=starttime, endtime=endtime, forcing='gswp3-w5e5', buffer=1)  # some buffer to avoid edge effects / errors in ISIMIP API
        rlds_coarse = self.download_isimip(variable='rlds', starttime=starttime, endtime=endtime, forcing='gswp3-w5e5', buffer=1)  # some buffer to avoid edge effects / errors in ISIMIP API
        
        regridder = xe.Regridder(hurs_coarse.isel(time=0).drop('time'), target, 'bilinear')

        hurs_coarse_regridded = regridder(hurs_coarse)
        tas_coarse_regridded = regridder(tas_coarse)
        rlds_coarse_regridded = regridder(rlds_coarse)

        hurs_fine = self.forcing['climate/hurs']
        tas_fine = self.forcing['climate/tas']

        output_coords = {}
        output_coords['time'] = pd.date_range(starttime, endtime, freq="D")    
        output_coords['y'] = self.grid.raster.coords['y']
        output_coords['x'] = self.grid.raster.coords['x']
        rlds_output = hydromt.raster.full(output_coords, dtype='float32', nodata=np.nan, crs=self.grid.raster.crs, name='rlds')

        # now ready for calculation:
        es_coarse = es0 * np.exp((lv / Rv) * (1 / T0 - 1 / tas_coarse_regridded.tas.data))  # saturation vapor pressure
        pV_coarse = (hurs_coarse_regridded.hurs.data * es_coarse) / 100  # water vapor pressure [hPa]

        es_fine = es0 * np.exp((lv / Rv) * (1 / T0 - 1 / tas_fine.data))
        pV_fine = (hurs_fine.data * es_fine) / 100  # water vapour pressure [hPa]

        e_cl_coarse = 0.23 + x1 * ((pV_coarse * 100) / tas_coarse_regridded.tas.data) ** (1 / x2)
        # e_cl_coarse == clear-sky emissivity w5e5 (pV needs to be in Pa not hPa, hence *100)
        e_cl_fine = 0.23 + x1 * ((pV_fine * 100) / tas_fine.data) ** (1 / x2)
        # e_cl_fine == clear-sky emissivity target grid (pV needs to be in Pa not hPa, hence *100)

        e_as_coarse = rlds_coarse_regridded.rlds.data / (sbc * tas_coarse_regridded.tas.data ** 4)  # all-sky emissivity w5e5
        e_as_coarse[e_as_coarse > 1] = 1  # constrain all-sky emissivity to max 1
        delta_e = e_as_coarse - e_cl_coarse  # cloud-based component of emissivity w5e5
        
        e_as_fine = e_cl_fine + delta_e
        e_as_fine[e_as_fine > 1] = 1  # constrain all-sky emissivity to max 1
        lw_fine = e_as_fine * sbc * tas_fine.data ** 4  # downscaled lwr! assume cloud e is the same

        rlds_output.data = lw_fine
        self.set_forcing(rlds_output, name='climate/rlds')

    def setup_pressure(self, starttime, endtime):
        g = 9.80665  # gravitational acceleration [m/s2]
        M = 0.02896968  # molar mass of dry air [kg/mol]
        r0 = 8.314462618  # universal gas constant [J/(mol·K)]
        T0 = 288.16  # Sea level standard temperature  [K]

        DEM = self.grid['landsurface/topo/elevation']
        pressure_30_min = self.download_isimip(variable='ps', starttime=starttime, endtime=endtime, forcing='gswp3-w5e5', buffer=1)['ps']  # some buffer to avoid edge effects / errors in ISIMIP API

        regridder = xe.Regridder(pressure_30_min.isel(time=0).drop('time'), DEM.rename({'x': 'lon', 'y': 'lat'}), 'bilinear')
 
        pressure_30_min_regridded = regridder(pressure_30_min)
        pressure_30_min_regridded_corr = pressure_30_min_regridded * np.exp(-(g * DEM.values * M) / (T0 * r0))

        output_coords = {}
        output_coords['time'] = pd.date_range(starttime, endtime, freq="D")    
        output_coords['y'] = self.grid.raster.coords['y']
        output_coords['x'] = self.grid.raster.coords['x']
        ps_output = hydromt.raster.full(output_coords, dtype='float32', nodata=np.nan, crs=self.grid.raster.crs, name='ps')
        ps_output.data = pressure_30_min_regridded_corr

        self.set_forcing(ps_output, name='climate/ps')

    def setup_high_resolution_variables(self, variables, starttime, endtime):
        for variable in variables:
            ds = self.download_isimip(variable=variable, starttime=starttime, endtime=endtime, forcing='chelsa-w5e5v1.0', resolution='30arcsec')
            var = ds[variable].raster.clip_bbox(ds.raster.bounds)
            self.set_forcing(var, name=f'climate/{variable}')

    def setup_hurs(self, starttime, endtime):
        # hurs
        hurs_30_min = self.download_isimip(variable='hurs', starttime=starttime, endtime=endtime, forcing='gswp3-w5e5', buffer=1)  # some buffer to avoid edge effects / errors in ISIMIP API

        # just taking the years to simplify things
        start_year = starttime.year
        end_year = endtime.year

        folder = Path(self.root).parent / 'preprocessing' / 'climate' / 'chelsa-bioclim+' / 'hurs'
        folder.mkdir(parents=True, exist_ok=True)

        hurs_ds_30sec, hurs_time = [], []
        for year in tqdm(range(start_year, end_year+1)):
            for month in range(1, 13):
                fn = folder / f'hurs_{year}_{month:02d}.nc'
                if not fn.exists():
                    hurs = self.data_catalog.get_rasterdataset(f'CHELSA-BIOCLIM+_monthly_hurs_{month:02d}_{year}', bbox=hurs_30_min.raster.bounds, buffer=1)
                    del hurs.attrs['_FillValue']
                    hurs.name = 'hurs'
                    hurs.to_netcdf(fn)
                else:
                    hurs = xr.open_dataset(fn)['hurs']
                hurs_ds_30sec.append(hurs)
                hurs_time.append(f'{year}-{month:02d}')
        
        hurs_ds_30sec = xr.concat(hurs_ds_30sec, dim='time').rename({'x': 'lon', 'y': 'lat'})
        hurs_ds_30sec.rio.set_spatial_dims('lon', 'lat', inplace=True)
        hurs_ds_30sec['time'] = pd.date_range(hurs_time[0], hurs_time[-1], freq="MS")

        output_coords = {}
        output_coords['time'] = pd.date_range(starttime, endtime, freq="D")    
        output_coords['y'] = self.grid.raster.coords['y']
        output_coords['x'] = self.grid.raster.coords['x']
        hurs_output = hydromt.raster.full(output_coords, name='hurs', dtype='float32', nodata=np.nan, crs=self.grid.raster.crs)

        regridder = xe.Regridder(hurs_30_min.isel(time=0).drop('time'), hurs_ds_30sec.isel(time=0).drop('time'), "bilinear")
        for year in tqdm(range(start_year, end_year+1)):
            for month in range(1, 13):
                start_month = datetime(year, month, 1)
                end_month = datetime(year, month, monthrange(year, month)[1])
                
                w5e5_30min_sel = hurs_30_min.sel(time=slice(start_month, end_month))
                w5e5_regridded = regridder(w5e5_30min_sel) * 0.01  # convert to fraction
                w5e5_regridded_mean = w5e5_regridded.mean(dim='time')  # get monthly mean
                w5e5_regridded_tr = np.log(w5e5_regridded / (1 - w5e5_regridded))  # assume beta distribuation => logit transform
                w5e5_regridded_mean_tr = np.log(w5e5_regridded_mean / (1 - w5e5_regridded_mean))  # logit transform

                chelsa = hurs_ds_30sec.sel(time=start_month) * 0.01  # convert to fraction
                chelsa_tr = np.log(chelsa / (1 - chelsa))  # assume beta distribuation => logit transform

                difference = chelsa_tr - w5e5_regridded_mean_tr

                # apply difference to w5e5
                w5e5_regridded_tr_corr = w5e5_regridded_tr + difference
                w5e5_regridded_corr = (1 / (1 + np.exp(-w5e5_regridded_tr_corr))) * 100  # back transform
                w5e5_regridded_corr.raster.set_crs(4326)

                hurs_output.loc[
                    dict(time=slice(start_month, end_month))
                ] = w5e5_regridded_corr['hurs'].raster.clip_bbox(hurs_output.raster.bounds)

        self.set_forcing(hurs_output, 'climate/hurs')

    def setup_wind(self, starttime, endtime):
        global_wind_atlas = rxr.open_rasterio(self.data_catalog['global_wind_atlas'].path).rio.clip_box(*self.grid.raster.bounds)
        # TODO: Gives memory errors when loading from disk.
        # global_wind_atlas = self.data_catalog.get_rasterdataset(
        #     'global_wind_atlas', bbox=self.grid.raster.bounds, buffer=10
        # ).rename({'x': 'lon', 'y': 'lat'})
        global_wind_atlas = global_wind_atlas.rename({'x': 'lon', 'y': 'lat'})
        target = self.grid['areamaps/grid_mask'].rename({'x': 'lon', 'y': 'lat'})
        regridder = xe.Regridder(global_wind_atlas.copy(), target, "bilinear")
        global_wind_atlas_regridded = regridder(global_wind_atlas)

        wind_30_min_avg = self.download_isimip(
            variable='sfcwind',
            starttime=datetime(2008, 1, 1),
            endtime=datetime(2017, 12, 31),
            forcing='gswp3-w5e5',
            buffer=1
        ).mean(dim='time')  # some buffer to avoid edge effects / errors in ISIMIP API
        regridder_30_min = xe.Regridder(wind_30_min_avg, target, "bilinear")
        wind_30_min_avg_regridded = regridder_30_min(wind_30_min_avg)

        # create diff layer:
        # assume wind follows weibull distribution => do log transform
        wind_30_min_avg_regridded_log = np.log(
            wind_30_min_avg_regridded.sfcwind.values,
            out=np.zeros_like(wind_30_min_avg_regridded.sfcwind.values),
            where=(wind_30_min_avg_regridded.sfcwind.values != 0)
        )  # avoid  -inf at locations where wind == 0

        global_wind_atlas_regridded_log = np.log(global_wind_atlas_regridded.values,
            out=np.zeros_like(global_wind_atlas_regridded.values),
            where=(global_wind_atlas_regridded.values != 0)
        )  # avoid  -inf at locations where wind == 0

        diff_layer = global_wind_atlas_regridded_log - wind_30_min_avg_regridded_log   # to be added to log-transformed daily

        wind_30_min = self.download_isimip(variable='sfcwind', starttime=starttime, endtime=endtime, forcing='gswp3-w5e5', buffer=1)  # some buffer to avoid edge effects / errors in ISIMIP API

        # just taking the years to simplify things
        start_year = starttime.year
        end_year = endtime.year

        output_coords = {}
        output_coords['time'] = pd.date_range(starttime, endtime, freq="D")    
        output_coords['y'] = self.grid.raster.coords['y']
        output_coords['x'] = self.grid.raster.coords['x']
        wind_output = hydromt.raster.full(output_coords, name='wind', dtype='float32', nodata=np.nan, crs=self.grid.raster.crs)

        for year in tqdm(range(start_year, end_year+1)):
            for month in range(1, 13):
                start_month = datetime(year, month, 1)
                end_month = datetime(year, month, monthrange(year, month)[1])
                
                wind_30min_sel = wind_30_min.sel(time=slice(start_month, end_month))
                wind_30min_sel_regridded = regridder_30_min(wind_30min_sel)

                wind_30min_sel_regridded_log = np.log(wind_30min_sel_regridded.sfcwind.values,
                    out=np.zeros_like(wind_30min_sel_regridded.sfcwind.values),
                    where=(wind_30min_sel_regridded.sfcwind.values != 0)  # avoid -inf at x=0
                )
                wind_30min_sel_regridded_log_corr = wind_30min_sel_regridded_log + diff_layer
                wind_30min_sel_regridded_corr = np.exp(
                    wind_30min_sel_regridded_log_corr,
                    out=np.zeros_like(wind_30min_sel_regridded_log_corr),
                    where=(wind_30min_sel_regridded_log_corr != 0)
                )

                wind_30min_sel_regridded_corr_ds = wind_30min_sel_regridded.copy()
                wind_30min_sel_regridded_corr_ds.sfcwind.values = wind_30min_sel_regridded_corr

                wind_output.loc[
                    dict(time=slice(start_month, end_month))
                ] = wind_30min_sel_regridded_corr_ds.sfcwind.raster.clip_bbox(wind_output.raster.bounds)

        self.set_forcing(wind_output, 'climate/wind')

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

    def add_grid_to_model_structure(self, grid: xr.Dataset, name: str) -> None:
        for var_name in grid.data_vars:
            self.model_structure[name][var_name] = var_name + '.tif'
        
    def write_grid(
        self,
        driver="GTiff",
        compress="deflate",
        **kwargs,
    ) -> None:
        self._assert_write_mode
        self.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
        self.add_grid_to_model_structure(self.grid, 'grid')
        if len(self.subgrid._grid) > 0:
            self.subgrid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
            self.add_grid_to_model_structure(self.subgrid.grid, 'subgrid')
        if len(self.region_subgrid._grid) > 0:
            self.region_subgrid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
            self.add_grid_to_model_structure(self.region_subgrid.grid, 'region_subgrid')
        if len(self.MERIT_grid._grid) > 0:
            self.MERIT_grid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
            self.add_grid_to_model_structure(self.MERIT_grid.grid, 'MERIT_grid')
        if len(self.MODFLOW_grid._grid) > 0:
            self.MODFLOW_grid.grid.raster.to_mapstack(self.root, driver=driver, compress=compress, **kwargs)
            self.add_grid_to_model_structure(self.MODFLOW_grid.grid, 'MODFLOW_grid')

    def write_forcing(self) -> None:
        self._assert_write_mode
        self.logger.info("Write forcing files")
        for var in self.forcing:
            forcing = self.forcing[var]
            fn = var + '.nc'
            self.model_structure['forcing'][var] = fn
            fp = Path(self.root, fn)
            fp.parent.mkdir(parents=True, exist_ok=True)
            forcing.to_netcdf(fp, mode='w')

    def write_table(self):
        if len(self.table) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.table.items():
                fn = os.path.join(name + '.csv')
                self.model_structure['table'][name] = fn
                self.logger.debug(f"Writing file {fn}")
                data.to_csv(os.path.join(self.root, fn))

    def write_binary(self):
        if len(self.binary) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.binary.items():
                fn = os.path.join(name + '.npz')
                self.model_structure['binary'][name] = fn
                self.logger.debug(f"Writing file {fn}")
                np.savez_compressed(os.path.join(self.root, fn), data=data)

    def write_dict(self):
        if len(self.dict) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.dict.items():
                fn = os.path.join(name + '.json')
                self.model_structure['dict'][name] = fn
                self.logger.debug(f"Writing file {fn}")
                output_path = Path(self.root) / fn
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f)

    def write_geoms(self, fn: str = "{name}.geojson", **kwargs) -> None:
        """Write model geometries to a vector file (by default GeoJSON) at <root>/<fn>

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'geoms/{name}.geojson'
        """
        if len(self._geoms) == 0:
            self.logger.debug("No geoms data found, skip writing.")
            return
        self._assert_write_mode
        if "driver" not in kwargs:
            kwargs.update(driver="GeoJSON")
        for name, gdf in self._geoms.items():
            self.logger.debug(f"Writing file {fn.format(name=name)}")
            self.model_structure["geoms"][name] = fn.format(name=name)
            _fn = os.path.join(self.root, fn.format(name=name))
            if not os.path.isdir(os.path.dirname(_fn)):
                os.makedirs(os.path.dirname(_fn))
            gdf.to_file(_fn, **kwargs)

    def set_table(self, table, name):
        self.table[name] = table

    def set_binary(self, data, name):
        self.binary[name] = data

    def set_dict(self, data, name):
        self.dict[name] = data

    def write_model_structure(self):
        with open(Path(self.root, "model_structure.json"), "w") as f:
            json.dump(self.model_structure, f, indent=4)

    def write(self):
        self.write_geoms()
        self.write_forcing()
        self.write_grid()
        self.write_table()
        self.write_binary()
        self.write_dict()

        self.write_model_structure()

    def read_model_structure(self):
        with open(Path(self.root, "model_structure.json"), "r") as f:
            self.model_structure = json.load(f)

    def read_geoms(self):
        for name, fn in self.model_structure["geoms"].items():
            self._geoms[name] = gpd.read_file(Path(self.root, fn))

    def read_binary(self):
        for name, fn in self.model_structure["binary"].items():
            self.binary[name] = np.load(Path(self.root, fn))["data"]
    
    def read_table(self):
        for name, fn in self.model_structure["table"].items():
            self.table[name] = pd.read_csv(Path(self.root, fn))

    def read_dict(self):
        for name, fn in self.model_structure["dict"].items():
            with open(Path(self.root, fn), "r") as f:
                self.dict[name] = json.load(f)

    def read_grid_from_disk(self, grid, name: str) -> None:
        data_arrays = []
        for name, fn in self.model_structure[name].items():
            with xr.load_dataset(Path(self.root) / fn, decode_cf=False).rename({'band_data': name}) as da:
                data_arrays.append(da.load())
            # with xr.load_dataarray(Path(self.root) / fn, decode_cf=False) as da:
            #     data_arrays.append(da.rename(name))
        ds = xr.merge(data_arrays)
        grid.set_grid(ds)

    def read_grid(self) -> None:
        self.read_grid_from_disk(self, 'grid')
        self.read_grid_from_disk(self.subgrid, 'subgrid')
        self.read_grid_from_disk(self.region_subgrid, 'region_subgrid')
        self.read_grid_from_disk(self.MERIT_grid, 'MERIT_grid')
        self.read_grid_from_disk(self.MODFLOW_grid, 'MODFLOW_grid')

    def read(self):
        self.read_model_structure()
        
        self.read_geoms()
        self.read_binary()
        self.read_table()
        self.read_dict()
        self.read_grid()

        # self.read_forcing()
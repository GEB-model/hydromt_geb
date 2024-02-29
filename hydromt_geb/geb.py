from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
import hydromt.workflows
from datetime import date, datetime
from typing import Union, Any, Dict
import logging
import os
import math
import requests
import time
import random
import zipfile
import json
from urllib.parse import urlparse
import concurrent.futures
from hydromt.exceptions import NoDataException
from honeybees.library.raster import pixels_to_coords

import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from affine import Affine
import xarray as xr
from dask.diagnostics import ProgressBar
import xclim.indices as xci
from dateutil.relativedelta import relativedelta

from hydromt.models.model_grid import GridModel

XY_CHUNKSIZE = 350

# temporary fix for ESMF on Windows
if os.name == "nt":
    os.environ["ESMFMKFILE"] = str(
        Path(os.__file__).parent.parent / "Library" / "lib" / "esmf.mk"
    )
else:
    os.environ["ESMFMKFILE"] = str(Path(os.__file__).parent.parent / "esmf.mk")

from affine import Affine
import geopandas as gpd

# use pyogrio for substantial speedup reading and writing vector data
gpd.options.io_engine = "pyogrio"

from calendar import monthrange
from isimip_client.client import ISIMIPClient

from .workflows import (
    repeat_grid,
    clip_with_grid,
    get_modflow_transform_and_shape,
    create_indices,
    create_modflow_basin,
    pad_xy,
    create_farms,
    get_farm_distribution,
    calculate_cell_area,
    fetch_and_save,
)
from .workflows.population import generate_locations

logger = logging.getLogger(__name__)


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class GEBModel(GridModel):
    _CLI_ARGS = {"region": "setup_grid"}

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
        epsg=4326,
        data_provider: str = None,
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
        self.data_provider = data_provider

        self._subgrid = None
        self._region_subgrid = None
        self._MERIT_grid = None
        self._MODFLOW_grid = None

        self.table = {}
        self.binary = {}
        self.dict = {}

        self.model_structure = {
            "forcing": {},
            "geoms": {},
            "grid": {},
            "dict": {},
            "table": {},
            "binary": {},
            "subgrid": {},
            "region_subgrid": {},
            "MERIT_grid": {},
            "MODFLOW_grid": {},
        }
        self.is_updated = {
            "forcing": {},
            "geoms": {},
            "grid": {},
            "dict": {},
            "table": {},
            "binary": {},
            "subgrid": {},
            "region_subgrid": {},
            "MERIT_grid": {},
            "MODFLOW_grid": {},
        }

    @property
    def subgrid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._subgrid is None:
            self._subgrid = xr.Dataset()
            if self._read:
                self.read_subgrid()
        return self._subgrid

    @property
    def region_subgrid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._region_subgrid is None:
            self._region_subgrid = xr.Dataset()
            if self._read:
                self.read_region_subgrid()
        return self._region_subgrid

    @property
    def MERIT_grid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._MERIT_grid is None:
            self._MERIT_grid = xr.Dataset()
            if self._read:
                self.read_MERIT_grid()
        return self._MERIT_grid

    @property
    def MODFLOW_grid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._MODFLOW_grid is None:
            self._MODFLOW_grid = xr.Dataset()
            if self._read:
                self.read_MODFLOW_grid()
        return self._MODFLOW_grid

    def setup_grid(
        self,
        region: dict,
        sub_grid_factor: int,
        hydrography_fn: str,
        basin_index_fn: str,
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

            Region must be of kind [basin, subbasin].
        sub_grid_factor : int
            GEB implements a subgrid. This parameter determines the factor by which the subgrid is smaller than the original grid.
        hydrography_fn : str
            Name of data source for hydrography data.
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`.
        """

        assert (
            sub_grid_factor > 10
        ), "sub_grid_factor must be larger than 10, because this is the resolution of the MERIT high-res DEM"
        assert sub_grid_factor % 10 == 0, "sub_grid_factor must be a multiple of 10"

        self.logger.info(f"Preparing 2D grid.")
        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        if kind in ["basin", "subbasin"]:
            # retrieve global hydrography data (lazy!)
            ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
            ds_org.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
            ds_org.y.attrs = {"long_name": "latitude", "units": "degrees_north"}
            if "bounds" not in region:
                region.update(
                    basin_index=self.data_catalog.get_geodataframe(basin_index_fn)
                )
            # get basin geometry
            geom, xy = hydromt.workflows.get_basin_geometry(
                ds=ds_org, kind=kind, logger=self.logger, **region
            )
            region.update(xy=xy)
            ds_hydro = ds_org.raster.clip_geom(geom, mask=True)
        else:
            raise ValueError(
                f"Region for grid must of kind [basin, subbasin], kind {kind} not understood."
            )

        # Add region and grid to model
        self.set_geoms(geom, name="areamaps/region")

        ldd = ds_hydro["flwdir"].raster.reclassify(
            reclass_table=pd.DataFrame(
                index=[
                    0,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    ds_hydro["flwdir"].raster.nodata,
                ],
                data={"ldd": [5, 6, 3, 2, 1, 4, 7, 8, 9, 0]},
            ),
            method="exact",
        )["ldd"]

        self.set_grid(ldd, name="routing/kinematic/ldd")
        self.set_grid(ds_hydro["uparea"], name="routing/kinematic/upstream_area")
        self.set_grid(ds_hydro["elevtn"], name="landsurface/topo/elevation")
        self.set_grid(
            xr.where(
                ds_hydro["rivlen_ds"] != -9999,
                ds_hydro["rivlen_ds"],
                np.nan,
                keep_attrs=True,
            ),
            name="routing/kinematic/channel_length",
        )
        self.set_grid(ds_hydro["rivslp"], name="routing/kinematic/channel_slope")

        # ds_hydro['mask'].raster.set_nodata(-1)
        self.set_grid((~ds_hydro["mask"]).astype(np.int8), name="areamaps/grid_mask")

        mask = self.grid["areamaps/grid_mask"]

        dst_transform = mask.raster.transform * Affine.scale(1 / sub_grid_factor)

        submask = hydromt.raster.full_from_transform(
            dst_transform,
            (
                mask.raster.shape[0] * sub_grid_factor,
                mask.raster.shape[1] * sub_grid_factor,
            ),
            nodata=0,
            dtype=mask.dtype,
            crs=mask.raster.crs,
            name="areamaps/sub_grid_mask",
            lazy=True,
        )
        submask.raster.set_nodata(None)
        submask.data = repeat_grid(mask.data, sub_grid_factor)

        self.set_subgrid(submask, name=submask.name)

    def setup_cell_area_map(self) -> None:
        """
        Sets up the cell area map for the model.

        Raises
        ------
        ValueError
            If the grid mask is not available.

        Notes
        -----
        This method prepares the cell area map for the model by calculating the area of each cell in the grid. It first
        retrieves the grid mask from the `areamaps/grid_mask` attribute of the grid, and then calculates the cell area
        using the `calculate_cell_area()` function. The resulting cell area map is then set as the `areamaps/cell_area`
        attribute of the grid.

        Additionally, this method sets up a subgrid for the cell area map by creating a new grid with the same extent as
        the subgrid, and then repeating the cell area values from the main grid to the subgrid using the `repeat_grid()`
        function, and correcting for the subgrid factor. Thus, every subgrid cell within a grid cell has the same value.
        The resulting subgrid cell area map is then set as the `areamaps/sub_cell_area` attribute of the subgrid.
        """
        self.logger.info(f"Preparing cell area map.")
        mask = self.grid["areamaps/grid_mask"].raster
        affine = mask.transform

        cell_area = hydromt.raster.full(
            mask.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="areamaps/cell_area",
            lazy=True,
        )
        cell_area.data = calculate_cell_area(affine, mask.shape)
        self.set_grid(cell_area, name=cell_area.name)

        sub_cell_area = hydromt.raster.full(
            self.subgrid.raster.coords,
            nodata=cell_area.raster.nodata,
            dtype=cell_area.dtype,
            name="areamaps/sub_cell_area",
            lazy=True,
        )

        sub_cell_area.data = (
            repeat_grid(cell_area.data, self.subgrid_factor) / self.subgrid_factor**2
        )
        self.set_subgrid(sub_cell_area, sub_cell_area.name)

    def setup_crops(
        self,
        crop_ids: dict,
        crop_variables: dict,
        crop_prices: Optional[Union[str, Dict[str, Any]]] = None,
        cultivation_costs: Optional[Union[str, Dict[str, Any]]] = None,
        project_future_until_year: Optional[int] = False,
    ):
        """
        Sets up the crops data for the model.

        Parameters
        ----------
        crop_ids : dict
            A dictionary of crop IDs and names.
        crop_variables : dict
            A dictionary of crop variables and their values.
        crop_prices : str or dict, optional
            The file path or dictionary of crop prices. If a file path is provided, the file is loaded and parsed as JSON.
            The dictionary should have a 'time' key with a list of time steps, and a 'crops' key with a dictionary of crop
            IDs and their prices.
        cultivation_costs : str or dict, optional
            The file path or dictionary of cultivation costs. If a file path is provided, the file is loaded and parsed as
            JSON. The dictionary should have a 'time' key with a list of time steps, and a 'crops' key with a dictionary of
            crop IDs and their cultivation costs.
        """
        self.logger.info(f"Preparing crops data")
        self.set_dict(crop_ids, name="crops/crop_ids")
        self.set_dict(crop_variables, name="crops/crop_variables")

        def project_to_future(df, project_future_until_year, inflation_rates):
            # expand table until year
            assert isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex)
            future_index = pd.date_range(
                df.index[-1],
                date(project_future_until_year, 12, 31),
                freq=pd.infer_freq(df.index),
                inclusive="right",
            )
            df = df.reindex(df.index.union(future_index))
            for future_date in future_index:
                source_date = future_date - pd.DateOffset(years=1)  # source is year ago
                inflation_index = inflation_rates["time"].index(str(future_date.year))
                for region_id, _ in df.columns:
                    region_inflation_rate = inflation_rates["data"][region_id][
                        inflation_index
                    ]
                    df.loc[future_date, region_id] = (
                        df.loc[source_date, region_id] * region_inflation_rate
                    ).values
            return df

        if crop_prices is not None:
            self.logger.info(f"Preparing crop prices")
            if isinstance(crop_prices, str):
                fp = Path(self.root, crop_prices)
                if not fp.exists():
                    raise ValueError(f"crop_prices file {fp.resolve()} does not exist")
                with open(fp, "r") as f:
                    crop_prices_data = json.load(f)
                crop_prices = pd.DataFrame(
                    {
                        crop_id: crop_prices_data["crops"][crop_name]
                        for crop_id, crop_name in crop_ids.items()
                    },
                    index=pd.to_datetime(crop_prices_data["time"]),
                )
                crop_prices = crop_prices.reindex(
                    columns=pd.MultiIndex.from_product(
                        [
                            self.geoms["areamaps/regions"]["region_id"],
                            crop_prices.columns,
                        ]
                    ),
                    level=1,
                )
                if project_future_until_year:
                    crop_prices = project_to_future(
                        crop_prices,
                        project_future_until_year,
                        self.dict["economics/inflation_rates"],
                    )

            crop_prices = {
                "time": crop_prices.index.tolist(),
                "data": {
                    str(region_id): crop_prices[region_id].to_dict(orient="list")
                    for region_id in self.geoms["areamaps/regions"]["region_id"]
                },
            }

            self.set_dict(crop_prices, name="crops/crop_prices")

        if cultivation_costs is not None:
            self.logger.info(f"Preparing cultivation costs")
            if isinstance(cultivation_costs, str):
                fp = Path(self.root, cultivation_costs)
                if not fp.exists():
                    raise ValueError(
                        f"cultivation_costs file {fp.resolve()} does not exist"
                    )
                with open(fp) as f:
                    cultivation_costs = json.load(f)
                cultivation_costs = pd.DataFrame(
                    {
                        crop_id: cultivation_costs["crops"][crop_name]
                        for crop_id, crop_name in crop_ids.items()
                    },
                    index=pd.to_datetime(cultivation_costs["time"]),
                )
                cultivation_costs = cultivation_costs.reindex(
                    columns=pd.MultiIndex.from_product(
                        [
                            self.geoms["areamaps/regions"]["region_id"],
                            cultivation_costs.columns,
                        ]
                    ),
                    level=1,
                )
                if project_future_until_year:
                    cultivation_costs = project_to_future(
                        cultivation_costs,
                        project_future_until_year,
                        self.dict["economics/inflation_rates"],
                    )
            cultivation_costs = {
                "time": cultivation_costs.index.tolist(),
                "data": {
                    str(region_id): cultivation_costs[region_id].to_dict(orient="list")
                    for region_id in self.geoms["areamaps/regions"]["region_id"]
                },
            }
            self.set_dict(cultivation_costs, name="crops/cultivation_costs")

    def setup_mannings(self) -> None:
        """
        Sets up the Manning's coefficient for the model.

        Notes
        -----
        This method sets up the Manning's coefficient for the model by calculating the coefficient based on the cell area
        and topography of the grid. It first calculates the upstream area of each cell in the grid using the
        `routing/kinematic/upstream_area` attribute of the grid. It then calculates the coefficient using the formula:

            C = 0.025 + 0.015 * (2 * A / U) + 0.030 * (Z / 2000)

        where C is the Manning's coefficient, A is the cell area, U is the upstream area, and Z is the elevation of the cell.

        The resulting Manning's coefficient is then set as the `routing/kinematic/mannings` attribute of the grid using the
        `set_grid()` method.
        """
        self.logger.info("Setting up Manning's coefficient")
        a = (2 * self.grid["areamaps/cell_area"]) / self.grid[
            "routing/kinematic/upstream_area"
        ]
        a = xr.where(a < 1, a, 1, keep_attrs=True)
        b = self.grid["landsurface/topo/elevation"] / 2000
        b = xr.where(b < 1, b, 1, keep_attrs=True)

        mannings = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="routing/kinematic/mannings",
            lazy=True,
        )
        mannings.data = 0.025 + 0.015 * a + 0.030 * b
        self.set_grid(mannings, mannings.name)

    def setup_channel_width(self, minimum_width: float) -> None:
        """
        Sets up the channel width for the model.

        Parameters
        ----------
        minimum_width : float
            The minimum channel width in meters.

        Notes
        -----
        This method sets up the channel width for the model by calculating the width of each channel based on the upstream
        area of each cell in the grid. It first retrieves the upstream area of each cell from the `routing/kinematic/upstream_area`
        attribute of the grid, and then calculates the channel width using the formula:

            W = A / 500

        where W is the channel width, and A is the upstream area of the cell. The resulting channel width is then set as
        the `routing/kinematic/channel_width` attribute of the grid using the `set_grid()` method.

        Additionally, this method sets a minimum channel width by replacing any channel width values that are less than the
        minimum width with the minimum width.
        """
        self.logger.info("Setting up channel width")
        channel_width_data = self.grid["routing/kinematic/upstream_area"] / 500
        channel_width_data = xr.where(
            channel_width_data > minimum_width,
            channel_width_data,
            minimum_width,
            keep_attrs=True,
        )

        channel_width = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="routing/kinematic/channel_width",
            lazy=True,
        )
        channel_width.data = channel_width_data

        self.set_grid(channel_width, channel_width.name)

    def setup_channel_depth(self) -> None:
        """
        Sets up the channel depth for the model.

        Raises
        ------
        AssertionError
            If the upstream area of any cell in the grid is less than or equal to zero.

        Notes
        -----
        This method sets up the channel depth for the model by calculating the depth of each channel based on the upstream
        area of each cell in the grid. It first retrieves the upstream area of each cell from the `routing/kinematic/upstream_area`
        attribute of the grid, and then calculates the channel depth using the formula:

            D = 0.27 * A ** 0.26

        where D is the channel depth, and A is the upstream area of the cell. The resulting channel depth is then set as
        the `routing/kinematic/channel_depth` attribute of the grid using the `set_grid()` method.

        Additionally, this method raises an `AssertionError` if the upstream area of any cell in the grid is less than or
        equal to zero. This is done to ensure that the upstream area is a positive value, which is required for the channel
        depth calculation to be valid.
        """
        self.logger.info("Setting up channel depth")
        assert (
            (self.grid["routing/kinematic/upstream_area"] > 0) | ~self.grid.mask
        ).all()
        channel_depth_data = 0.27 * self.grid["routing/kinematic/upstream_area"] ** 0.26
        channel_depth = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="routing/kinematic/channel_depth",
            lazy=True,
        )
        channel_depth.data = channel_depth_data
        self.set_grid(channel_depth, channel_depth.name)

    def setup_channel_ratio(self) -> None:
        """
        Sets up the channel ratio for the model.

        Raises
        ------
        AssertionError
            If the channel length of any cell in the grid is less than or equal to zero, or if the channel ratio of any
            cell in the grid is less than zero.

        Notes
        -----
        This method sets up the channel ratio for the model by calculating the ratio of the channel area to the cell area
        for each cell in the grid. It first retrieves the channel width and length from the `routing/kinematic/channel_width`
        and `routing/kinematic/channel_length` attributes of the grid, and then calculates the channel area using the
        product of the width and length. It then calculates the channel ratio by dividing the channel area by the cell area
        retrieved from the `areamaps/cell_area` attribute of the grid.

        The resulting channel ratio is then set as the `routing/kinematic/channel_ratio` attribute of the grid using the
        `set_grid()` method. Any channel ratio values that are greater than 1 are replaced with 1 (i.e., the whole cell is a channel).

        Additionally, this method raises an `AssertionError` if the channel length of any cell in the grid is less than or
        equal to zero, or if the channel ratio of any cell in the grid is less than zero. These checks are done to ensure
        that the channel length and ratio are positive values, which are required for the channel ratio calculation to be
        valid.
        """
        self.logger.info("Setting up channel ratio")
        assert (
            (self.grid["routing/kinematic/channel_length"] > 0) | ~self.grid.mask
        ).all()
        channel_area = (
            self.grid["routing/kinematic/channel_width"]
            * self.grid["routing/kinematic/channel_length"]
        )
        channel_ratio_data = channel_area / self.grid["areamaps/cell_area"]
        channel_ratio_data = xr.where(
            channel_ratio_data < 1, channel_ratio_data, 1, keep_attrs=True
        )
        assert ((channel_ratio_data >= 0) | ~self.grid.mask).all()
        channel_ratio = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="routing/kinematic/channel_ratio",
            lazy=True,
        )
        channel_ratio.data = channel_ratio_data
        self.set_grid(channel_ratio, channel_ratio.name)

    def setup_elevation_STD(self) -> None:
        """
        Sets up the standard deviation of elevation for the model.

        Notes
        -----
        This method sets up the standard deviation of elevation for the model by retrieving high-resolution elevation data
        from the MERIT dataset and calculating the standard deviation of elevation for each cell in the grid.

        MERIT data has a half cell offset. Therefore, this function first corrects for this offset.  It then selects the
        high-resolution elevation data from the MERIT dataset using the grid coordinates of the model, and calculates the
        standard deviation of elevation for each cell in the grid using the `np.std()` function.

        The resulting standard deviation of elevation is then set as the `landsurface/topo/elevation_STD` attribute of
        the grid using the `set_grid()` method.
        """
        self.logger.info("Setting up elevation standard deviation")
        MERIT = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            variables=["elv"],
            provider=self.data_provider,
            bbox=self.grid.raster.bounds,
            buffer=50,
        ).compute()  # Why is compute needed here?
        # In some MERIT datasets, there is a half degree offset in MERIT data. We can detect this by checking the offset relative to the resolution.
        # This offset should be 0.5. If the offset instead is close to 0 or 1, then we need to correct for this offset.
        center_offset = (
            MERIT.coords["x"][0] % MERIT.rio.resolution()[0]
        ) / MERIT.rio.resolution()[0]
        # check whether offset is close to 0.5
        if not np.isclose(center_offset, 0.5, atol=MERIT.rio.resolution()[0] / 100):
            assert np.isclose(
                center_offset, 0, atol=MERIT.rio.resolution()[0] / 100
            ) or np.isclose(
                center_offset, 1, atol=MERIT.rio.resolution()[0] / 100
            ), "Could not detect offset in MERIT data"
            MERIT = MERIT.assign_coords(
                x=MERIT.coords["x"] + MERIT.rio.resolution()[0] / 2,
                y=MERIT.coords["y"] - MERIT.rio.resolution()[1] / 2,
            )
            center_offset = (
                MERIT.coords["x"][0] % MERIT.rio.resolution()[0]
            ) / MERIT.rio.resolution()[0]

        # we are going to match the upper left corners. So create a MERIT grid with the upper left corners as coordinates
        MERIT_ul = MERIT.assign_coords(
            x=MERIT.coords["x"] - MERIT.rio.resolution()[0] / 2,
            y=MERIT.coords["y"] - MERIT.rio.resolution()[1] / 2,
        )

        scaling = 10

        # find the upper left corner of the grid cells in self.grid
        y_step = self.grid.get_index("y")[1] - self.grid.get_index("y")[0]
        x_step = self.grid.get_index("x")[1] - self.grid.get_index("x")[0]
        upper_left_y = self.grid.get_index("y")[0] - y_step / 2
        upper_left_x = self.grid.get_index("x")[0] - x_step / 2

        ymin = np.isclose(
            MERIT_ul.get_index("y"), upper_left_y, atol=MERIT.rio.resolution()[1] / 100
        )
        assert (
            ymin.sum() == 1
        ), "Could not find the upper left corner of the grid cell in MERIT data"
        ymin = ymin.argmax()
        ymax = ymin + self.grid.y.size * scaling
        xmin = np.isclose(
            MERIT_ul.get_index("x"), upper_left_x, atol=MERIT.rio.resolution()[0] / 100
        )
        assert (
            xmin.sum() == 1
        ), "Could not find the upper left corner of the grid cell in MERIT data"
        xmin = xmin.argmax()
        xmax = xmin + self.grid.x.size * scaling

        # select data from MERIT using the grid coordinates
        high_res_elevation_data = MERIT.isel(y=slice(ymin, ymax), x=slice(xmin, xmax))
        self.set_MERIT_grid(
            MERIT.isel(y=slice(ymin - 1, ymax + 1), x=slice(xmin - 1, xmax + 1)),
            name="landsurface/topo/subgrid_elevation",
        )

        elevation_per_cell = (
            high_res_elevation_data.values.reshape(
                high_res_elevation_data.shape[0] // scaling, scaling, -1, scaling
            )
            .swapaxes(1, 2)
            .reshape(-1, scaling, scaling)
        )

        elevation_per_cell = high_res_elevation_data.values.reshape(
            high_res_elevation_data.shape[0] // scaling, scaling, -1, scaling
        ).swapaxes(1, 2)

        standard_deviation = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="landsurface/topo/elevation_STD",
            lazy=True,
        )
        standard_deviation.data = np.std(elevation_per_cell, axis=(2, 3))
        self.set_grid(standard_deviation, standard_deviation.name)

    def setup_soil_parameters(self, interpolation_method="nearest") -> None:
        """
        Sets up the soil parameters for the model.

        Parameters
        ----------
        interpolation_method : str, optional
            The interpolation method to use when interpolating the soil parameters. Default is 'nearest'.

        Notes
        -----
        This method sets up the soil parameters for the model by retrieving soil data from the CWATM dataset and interpolating
        the data to the model grid. It first retrieves the soil dataset from the `data_catalog`, and
        then retrieves the soil parameters and storage depth data for each soil layer. It then interpolates the data to the
        model grid using the specified interpolation method and sets the resulting grids as attributes of the model.

        Additionally, this method sets up the percolation impeded and crop group data by retrieving the corresponding data
        from the soil dataset and interpolating it to the model grid.

        The resulting soil parameters are set as attributes of the model with names of the form 'soil/{parameter}{soil_layer}',
        where {parameter} is the name of the soil parameter (e.g. 'alpha', 'ksat', etc.) and {soil_layer} is the index of the
        soil layer (1-3; 1 is the top layer). The storage depth data is set as attributes of the model with names of the
        form 'soil/storage_depth{soil_layer}'. The percolation impeded and crop group data are set as attributes of the model
        with names 'soil/percolation_impeded' and 'soil/cropgrp', respectively.
        """
        self.logger.info("Setting up soil parameters")
        soil_ds = self.data_catalog.get_rasterdataset(
            "cwatm_soil_5min", bbox=self.bounds, buffer=10
        )
        for parameter in ("alpha", "ksat", "lambda", "thetar", "thetas"):
            for soil_layer in range(1, 4):
                ds = soil_ds[f"{parameter}{soil_layer}_5min"]
                self.set_grid(
                    self.interpolate(ds, interpolation_method),
                    name=f"soil/{parameter}{soil_layer}",
                )

        for soil_layer in range(1, 3):
            ds = soil_ds[f"storageDepth{soil_layer}"]
            self.set_grid(
                self.interpolate(ds, interpolation_method),
                name=f"soil/storage_depth{soil_layer}",
            )

        ds = soil_ds["percolationImp"]
        self.set_grid(
            self.interpolate(ds, interpolation_method), name=f"soil/percolation_impeded"
        )
        ds = soil_ds["cropgrp"]
        self.set_grid(self.interpolate(ds, interpolation_method), name=f"soil/cropgrp")

    def setup_land_use_parameters(self, interpolation_method="nearest") -> None:
        """
        Sets up the land use parameters for the model.

        Parameters
        ----------
        interpolation_method : str, optional
            The interpolation method to use when interpolating the land use parameters. Default is 'nearest'.

        Notes
        -----
        This method sets up the land use parameters for the model by retrieving land use data from the CWATM dataset and
        interpolating the data to the model grid. It first retrieves the land use dataset from the `data_catalog`, and
        then retrieves the maximum root depth and root fraction data for each land use type. It then
        interpolates the data to the model grid using the specified interpolation method and sets the resulting grids as
        attributes of the model with names of the form 'landcover/{land_use_type}/{parameter}_{land_use_type}', where
        {land_use_type} is the name of the land use type (e.g. 'forest', 'grassland', etc.) and {parameter} is the name of
        the land use parameter (e.g. 'maxRootDepth', 'rootFraction1', etc.).

        Additionally, this method sets up the crop coefficient and interception capacity data for each land use type by
        retrieving the corresponding data from the land use dataset and interpolating it to the model grid. The crop
        coefficient data is set as attributes of the model with names of the form 'landcover/{land_use_type}/cropCoefficient{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset. The interception capacity
        data is set as attributes of the model with names of the form 'landcover/{land_use_type}/interceptCap{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset.

        The resulting land use parameters are set as attributes of the model with names of the form 'landcover/{land_use_type}/{parameter}_{land_use_type}',
        where {land_use_type} is the name of the land use type (e.g. 'forest', 'grassland', etc.) and {parameter} is the name of
        the land use parameter (e.g. 'maxRootDepth', 'rootFraction1', etc.). The crop coefficient data is set as attributes
        of the model with names of the form 'landcover/{land_use_type}/cropCoefficient{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset. The interception capacity
        data is set as attributes of the model with names of the form 'landcover/{land_use_type}/interceptCap{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset.
        """
        self.logger.info("Setting up land use parameters")
        for land_use_type, land_use_type_netcdf_name in (
            ("forest", "Forest"),
            ("grassland", "Grassland"),
            ("irrPaddy", "irrPaddy"),
            ("irrNonPaddy", "irrNonPaddy"),
        ):
            self.logger.info(f"Setting up land use parameters for {land_use_type}")
            land_use_ds = self.data_catalog.get_rasterdataset(
                f"cwatm_{land_use_type}_5min", bbox=self.bounds, buffer=10
            )

            for parameter in ("maxRootDepth", "rootFraction1"):
                self.set_grid(
                    self.interpolate(land_use_ds[parameter], interpolation_method),
                    name=f"landcover/{land_use_type}/{parameter}_{land_use_type}",
                )

            parameter = f"cropCoefficient{land_use_type_netcdf_name}_10days"
            self.set_forcing(
                self.interpolate(land_use_ds[parameter], interpolation_method),
                name=f"landcover/{land_use_type}/{parameter}",
            )
            if land_use_type in ("forest", "grassland"):
                parameter = f"interceptCap{land_use_type_netcdf_name}_10days"
                self.set_forcing(
                    self.interpolate(land_use_ds[parameter], interpolation_method),
                    name=f"landcover/{land_use_type}/{parameter}",
                )

    def setup_waterbodies(
        self,
        command_areas="reservoir_command_areas",
        custom_reservoir_capacity="custom_reservoir_capacity",
    ):
        """
        Sets up the waterbodies for GEB.

        Notes
        -----
        This method sets up the waterbodies for GEB. It first retrieves the waterbody data from the
        specified data catalog and sets it as a geometry in the model. It then rasterizes the waterbody data onto the model
        grid and the subgrid using the `rasterize` method of the `raster` object. The resulting grids are set as attributes
        of the model with names of the form 'routing/lakesreservoirs/{grid_name}'.

        The method also retrieves the reservoir command area data from the data catalog and calculates the area of each
        command area that falls within the model region. The `waterbody_id` key is used to do the matching between these
        databases. The relative area of each command area within the model region is calculated and set as a column in
        the waterbody data. The method sets all lakes with a command area to be reservoirs and updates the waterbody data
        with any custom reservoir capacity data from the data catalog.

        TODO: Make the reservoir command area data optional.

        The resulting waterbody data is set as a table in the model with the name 'routing/lakesreservoirs/basin_lakes_data'.
        """
        self.logger.info("Setting up waterbodies")
        try:
            waterbodies = self.data_catalog.get_geodataframe(
                "hydro_lakes",
                geom=self.geoms["areamaps/region"],
                predicate="intersects",
                variables=[
                    "waterbody_id",
                    "waterbody_type",
                    "volume_total",
                    "average_discharge",
                    "average_area",
                ],
            )
        except (IndexError, NoDataException):
            self.logger.info(
                "No water bodies found in domain, skipping water bodies setup"
            )
            waterbodies = gpd.GeoDataFrame(
                columns=[
                    "waterbody_id",
                    "waterbody_type",
                    "volume_total",
                    "average_discharge",
                    "average_area",
                    "geometry",
                ],
                crs=self.crs,
            )
            lakesResID = xr.zeros_like(self.grid["areamaps/grid_mask"])
            sublakesResID = xr.zeros_like(self.subgrid["areamaps/sub_grid_mask"])

        else:
            lakesResID = self.grid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=0,
                all_touched=True,
                dtype=np.int32,
            )
            sublakesResID = self.subgrid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=0,
                all_touched=True,
                dtype=np.int32,
            )

        self.set_grid(lakesResID, name="routing/lakesreservoirs/lakesResID")
        self.set_subgrid(sublakesResID, name="routing/lakesreservoirs/sublakesResID")

        waterbodies = waterbodies.set_index("waterbody_id")
        waterbodies["volume_flood"] = waterbodies["volume_total"]

        if command_areas:
            command_areas = self.data_catalog.get_geodataframe(
                command_areas, geom=self.region, predicate="intersects"
            )
            command_areas = command_areas[
                ~command_areas["waterbody_id"].isnull()
            ].reset_index(drop=True)
            command_areas["waterbody_id"] = command_areas["waterbody_id"].astype(
                np.int32
            )
            command_areas["geometry_in_region_bounds"] = gpd.overlay(
                command_areas, self.region, how="intersection", keep_geom_type=False
            )["geometry"]
            command_areas["area"] = command_areas.to_crs(3857).area
            command_areas["area_in_region_bounds"] = (
                command_areas["geometry_in_region_bounds"].to_crs(3857).area
            )
            areas_per_waterbody = command_areas.groupby("waterbody_id").agg(
                {"area": "sum", "area_in_region_bounds": "sum"}
            )
            relative_area_in_region = (
                areas_per_waterbody["area_in_region_bounds"]
                / areas_per_waterbody["area"]
            )
            relative_area_in_region.name = (
                "relative_area_in_region"  # set name for merge
            )

            self.set_grid(
                self.grid.raster.rasterize(
                    command_areas,
                    col_name="waterbody_id",
                    nodata=-1,
                    all_touched=True,
                    dtype=np.int32,
                ),
                name="routing/lakesreservoirs/command_areas",
            )
            self.set_subgrid(
                self.subgrid.raster.rasterize(
                    command_areas,
                    col_name="waterbody_id",
                    nodata=-1,
                    all_touched=True,
                    dtype=np.int32,
                ),
                name="routing/lakesreservoirs/subcommand_areas",
            )

            # set all lakes with command area to reservoir
            waterbodies.loc[
                waterbodies.index.isin(command_areas["waterbody_id"]), "waterbody_type"
            ] = 2
            # set relative area in region for command area. If no command area, set this is set to nan.
            waterbodies = waterbodies.merge(
                relative_area_in_region, how="left", left_index=True, right_index=True
            )
        else:
            command_areas = hydromt.raster.full(
                self.grid.raster.coords,
                nodata=-1,
                dtype=np.int32,
                name="areamaps/sub_grid_mask",
                crs=self.grid.raster.crs,
                lazy=True,
            )
            self.set_grid(command_areas, name="routing/lakesreservoirs/command_areas")
            waterbodies["relative_area_in_region"] = 1

        if custom_reservoir_capacity:
            custom_reservoir_capacity = self.data_catalog.get_dataframe(
                "custom_reservoir_capacity"
            ).set_index("waterbody_id")
            custom_reservoir_capacity = custom_reservoir_capacity[
                custom_reservoir_capacity.index != -1
            ]

            waterbodies.update(custom_reservoir_capacity)

        # spatial dimension is not required anymore, so drop it.
        waterbodies = waterbodies.drop("geometry", axis=1)

        self.set_table(waterbodies, name="routing/lakesreservoirs/basin_lakes_data")

    def setup_water_demand(self, starttime, endtime, ssp):
        """
        Sets up the water demand data for GEB.

        Notes
        -----
        This method sets up the water demand data for GEB. It retrieves the domestic, industry, and
        livestock water demand data from the specified data catalog and sets it as forcing data in the model. The domestic
        water demand and consumption data are retrieved from the 'cwatm_domestic_water_demand' dataset, while the industry
        water demand and consumption data are retrieved from the 'cwatm_industry_water_demand' dataset. The livestock water
        consumption data is retrieved from the 'cwatm_livestock_water_demand' dataset.

        The domestic water demand and consumption data are provided at a monthly time step, while the industry water demand
        and consumption data are provided at an annual time step. The livestock water consumption data is provided at a
        monthly time step, but is assumed to be constant over the year.

        The resulting water demand data is set as forcing data in the model with names of the form 'water_demand/{demand_type}'.
        """
        self.logger.info("Setting up water demand")

        def set(file, accessor, name, ssp, starttime, endtime):
            ds_historic = self.data_catalog.get_rasterdataset(
                f"cwatm_{file}_historical", bbox=self.bounds, buffer=2
            )
            if accessor:
                ds_historic = getattr(ds_historic, accessor)
            ds_future = self.data_catalog.get_rasterdataset(
                f"cwatm_{file}_{ssp}", bbox=self.bounds, buffer=2
            )
            if accessor:
                ds_future = getattr(ds_future, accessor)
            ds = xr.concat([ds_historic, ds_future], dim="time")
            ds["time"] = pd.date_range(
                start=datetime(1901, 1, 1)
                + relativedelta(months=int(ds.time[0].data.item())),
                periods=len(ds.time),
                freq="AS",
            )
            assert (ds.time.dt.year.diff("time") == 1).all(), "not all years are there"
            ds = ds.sel(time=slice(starttime, endtime))
            ds.name = name
            self.set_forcing(
                ds.rename({"lat": "y", "lon": "x"}), name=f"water_demand/{name}"
            )

        set(
            "domestic_water_demand",
            "domWW",
            "domestic_water_demand",
            ssp,
            starttime,
            endtime,
        )
        set(
            "domestic_water_demand",
            "domCon",
            "domestic_water_consumption",
            ssp,
            starttime,
            endtime,
        )
        set(
            "industry_water_demand",
            "indWW",
            "industry_water_demand",
            ssp,
            starttime,
            endtime,
        )
        set(
            "industry_water_demand",
            "indCon",
            "industry_water_consumption",
            ssp,
            starttime,
            endtime,
        )
        set(
            "livestock_water_demand",
            None,
            "livestock_water_consumption",
            ssp,
            starttime,
            endtime,
        )

    def setup_modflow(self, epsg: int, resolution: float):
        """
        Sets up the MODFLOW grid for GEB.

        Parameters
        ----------
        epsg : int
            The EPSG code for the coordinate reference system of the model grid.
        resolution : float
            The resolution of the model grid in meters.

        Notes
        -----
        This method sets up the MODFLOW grid for GEB. These grids don't match because one is based on
        a geographic coordinate reference system and the other is based on a projected coordinate reference system. Therefore,
        this function creates a projected MODFLOW grid and then calculates the intersection between the model grid and the MODFLOW
        grid.

        It first retrieves the MODFLOW mask from the `get_modflow_transform_and_shape` function, which calculates the affine
        transform and shape of the MODFLOW grid based on the resolution and EPSG code of the model grid. The MODFLOW mask is
        created using the `full_from_transform` method of the `raster` object, which creates a binary grid with the same affine
        transform and shape as the MODFLOW grid.

        The method then creates an intersection between the model grid and the MODFLOW grid using the `create_indices`
        function. The resulting indices are used to match cells between the model grid and the MODFLOW grid. The indices
        are saved for use in the model.

        Finally, the elevation data for the MODFLOW grid is retrieved from the MERIT dataset and reprojected to the MODFLOW
        grid using the `reproject_like` method of the `raster` object. The resulting elevation grid is set as a grid in the
        model with the name 'groundwater/modflow/modflow_elevation'.
        """
        self.logger.info("Setting up MODFLOW")
        modflow_affine, MODFLOW_shape = get_modflow_transform_and_shape(
            self.grid["landsurface/topo/elevation"], 4326, epsg, resolution
        )
        modflow_mask = hydromt.raster.full_from_transform(
            modflow_affine,
            MODFLOW_shape,
            nodata=0,
            dtype=np.int8,
            name=f"groundwater/modflow/modflow_mask",
            crs=epsg,
            lazy=True,
        )

        intersection = create_indices(
            self.grid["landsurface/topo/elevation"].raster.transform,
            self.grid["landsurface/topo/elevation"].raster.shape,
            4326,
            modflow_affine,
            MODFLOW_shape,
            epsg,
        )

        self.set_binary(
            intersection["y_modflow"], name=f"groundwater/modflow/y_modflow"
        )
        self.set_binary(
            intersection["x_modflow"], name=f"groundwater/modflow/x_modflow"
        )
        self.set_binary(intersection["y_hydro"], name=f"groundwater/modflow/y_hydro")
        self.set_binary(intersection["x_hydro"], name=f"groundwater/modflow/x_hydro")
        self.set_binary(intersection["area"], name=f"groundwater/modflow/area")

        modflow_mask.data = create_modflow_basin(
            self.grid["landsurface/topo/elevation"], intersection, MODFLOW_shape
        )
        self.set_MODFLOW_grid(modflow_mask, name=f"groundwater/modflow/modflow_mask")

        MERIT = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            variables=["elv"],
            bbox=self.bounds,
            buffer=50,
            provider=self.data_provider,
        )
        MERIT_x_step = MERIT.coords["x"][1] - MERIT.coords["x"][0]
        MERIT_y_step = MERIT.coords["y"][0] - MERIT.coords["y"][1]
        MERIT = MERIT.assign_coords(
            x=MERIT.coords["x"] + MERIT_x_step / 2,
            y=MERIT.coords["y"] + MERIT_y_step / 2,
        )
        elevation_modflow = MERIT.raster.reproject_like(modflow_mask, method="average")

        self.set_MODFLOW_grid(
            elevation_modflow, name=f"groundwater/modflow/modflow_elevation"
        )

    def setup_forcing(
        self,
        starttime: date,
        endtime: date,
        data_source: str = "isimip",
        resolution_arcsec: int = 30,
        forcing: str = "chelsa-w5e5v1.0",
        ssp=None,
        calculate_SPEI: bool = True,
        calculate_GEV: bool = True,
    ):
        """
        Sets up the forcing data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        data_source : str, optional
            The data source to use for the forcing data. Default is 'isimip'.

        Notes
        -----
        This method sets up the forcing data for GEB. It first downloads the high-resolution variables
        (precipitation, surface solar radiation, air temperature, maximum air temperature, and minimum air temperature) from
        the ISIMIP dataset for the specified time period. The data is downloaded using the `setup_30arcsec_variables_isimip`
        method.

        The method then sets up the relative humidity, longwave radiation, pressure, and wind data for the model. The
        relative humidity data is downloaded from the ISIMIP dataset using the `setup_hurs_isimip_30arcsec` method. The longwave radiation
        data is calculated using the air temperature and relative humidity data and the `calculate_longwave` function. The
        pressure data is downloaded from the ISIMIP dataset using the `setup_pressure_isimip_30arcsec` method. The wind data is downloaded
        from the ISIMIP dataset using the `setup_wind_isimip_30arcsec` method. All these data are first downscaled to the model grid.

        The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.
        """
        assert starttime < endtime, "Start time must be before end time"

        if data_source == "isimip":
            if resolution_arcsec == 30:
                assert (
                    forcing == "chelsa-w5e5v1.0"
                ), "Only chelsa-w5e5v1.0 is supported for 30 arcsec resolution"
                # download source data from ISIMIP
                self.logger.info("setting up forcing data")
                high_res_variables = ["pr", "rsds", "tas", "tasmax", "tasmin"]
                self.setup_30arcsec_variables_isimip(
                    high_res_variables, starttime, endtime
                )
                self.logger.info("setting up relative humidity...")
                self.setup_hurs_isimip_30arcsec(starttime, endtime)
                self.logger.info("setting up longwave radiation...")
                self.setup_longwave_isimip_30arcsec(
                    starttime=starttime, endtime=endtime
                )
                self.logger.info("setting up pressure...")
                self.setup_pressure_isimip_30arcsec(starttime, endtime)
                self.logger.info("setting up wind...")
                self.setup_wind_isimip_30arcsec(starttime, endtime)
            elif resolution_arcsec == 1800:
                variables = [
                    "pr",
                    "rsds",
                    "tas",
                    "tasmax",
                    "tasmin",
                    "hurs",
                    "rlds",
                    "ps",
                    "sfcwind",
                ]
                self.setup_1800arcsec_variables_isimip(
                    forcing, variables, starttime, endtime, ssp=ssp
                )
        elif data_source == "era5":
            # # Create a thread pool and map the set_forcing function to the variables
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(self.download_ERA, variable, starttime, endtime) for variable in variables]

            # # Wait for all threads to complete
            # concurrent.futures.wait(futures)
            DEM = self.grid["landsurface/topo/elevation"]

            # ERA5_elevation = (
            #     (
            #         self.data_catalog.get_rasterdataset(
            #             "ERA5_geopotential", bbox=self.bounds, buffer=1
            #         )
            #         / 9.80665
            #     )
            #     .isel(time=0)
            #     .rename({"longitude": "x", "latitude": "y"})
            # )  # convert from m2/s2 to m (see: https://codes.ecmwf.int/grib/param-db/129)
            # # LAPSE_RATE = -0.0065

            import concurrent.futures

            variables = [
                "total_precipitation",
                "surface_solar_radiation_downwards",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
                "surface_thermal_radiation_downwards",
            ]
            import multiprocessing

            with multiprocessing.Pool() as pool:
                results = pool.starmap(
                    self.download_ERA,
                    [
                        (variable, starttime, endtime, None, True)
                        for variable in variables
                    ],
                )

            pr_hourly = self.download_ERA(
                "total_precipitation", starttime, endtime, method="accumulation"
            )
            pr_hourly = pr_hourly * (1000 / 3600)  # convert from m/hr to kg/m2/s
            # ensure no negative values for precipitation, which may arise due to float precision
            pr_hourly = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
            pr_hourly.name = "pr_hourly"
            self.set_forcing(pr_hourly, name="climate/pr_hourly")
            pr = pr_hourly.resample(time="D").mean()  # get daily mean
            pr = pr.raster.reproject_like(DEM, method="average")
            pr.name = "pr"
            self.set_forcing(pr, name="climate/pr")

            hourly_rsds = self.download_ERA(
                "surface_solar_radiation_downwards",
                starttime,
                endtime,
                method="accumulation",
            )
            rsds = hourly_rsds.resample(time="D").sum() / (
                24 * 3600
            )  # get daily sum and convert from J/m2 to W/m2
            rsds = rsds.raster.reproject_like(DEM, method="average")
            rsds.name = "rsds"
            self.set_forcing(rsds, name="climate/rsds")

            hourly_rlds = self.download_ERA(
                "surface_thermal_radiation_downwards",
                starttime,
                endtime,
                method="accumulation",
            )
            rlds = hourly_rlds.resample(time="D").sum() / (24 * 3600)
            rlds = rlds.raster.reproject_like(DEM, method="average")
            rlds.name = "rlds"
            self.set_forcing(rlds, name="climate/rlds")

            hourly_tas = self.download_ERA(
                "2m_temperature", starttime, endtime, method="raw"
            )
            tas = hourly_tas.resample(time="D").mean()
            # tas_sea_level = tas - (ERA5_elevation * LAPSE_RATE)
            tas_reprojected = tas.raster.reproject_like(DEM, method="average")
            tas_reprojected.name = "tas"
            self.set_forcing(tas_reprojected, name="climate/tas")

            tasmax = hourly_tas.resample(time="D").max()
            tasmax = tasmax.raster.reproject_like(DEM, method="average")
            tasmax.name = "tasmax"
            self.set_forcing(tasmax, name="climate/tasmax")

            tasmin = hourly_tas.resample(time="D").min()
            tasmin = tasmin.raster.reproject_like(DEM, method="average")
            tasmin.name = "tasmin"
            self.set_forcing(tasmin, name="climate/tasmin")

            dew_point_tas_C = (
                self.download_ERA(
                    "2m_dewpoint_temperature", starttime, endtime, method="raw"
                )
                - 273.15
            )
            hourly_tas_C = hourly_tas - 273.15
            water_vapour_pressure = 0.6108 * np.exp(
                17.27 * dew_point_tas_C / (237.3 + dew_point_tas_C)
            )  # calculate water vapour pressure (kPa)
            saturation_vapour_pressure = 0.6108 * np.exp(
                17.27 * hourly_tas_C / (237.3 + hourly_tas_C)
            )
            assert water_vapour_pressure.shape == saturation_vapour_pressure.shape
            relative_humidity = (
                water_vapour_pressure / saturation_vapour_pressure
            ) * 100
            relative_humidity = relative_humidity.resample(time="D").mean()
            relative_humidity = relative_humidity.raster.reproject_like(
                DEM, method="average"
            )
            relative_humidity.name = "hurs"
            self.set_forcing(relative_humidity, name="climate/hurs")

            pressure = self.download_ERA(
                "surface_pressure", starttime, endtime, method="raw"
            )
            pressure = pressure.resample(time="D").mean()
            pressure = pressure.raster.reproject_like(DEM, method="average")
            pressure.name = "ps"
            self.set_forcing(pressure, name="climate/ps")

            u_wind = self.download_ERA(
                "10m_u_component_of_wind", starttime, endtime, method="raw"
            )
            u_wind = u_wind.resample(time="D").mean()

            v_wind = self.download_ERA(
                "10m_v_component_of_wind", starttime, endtime, method="raw"
            )
            v_wind = v_wind.resample(time="D").mean()
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)
            wind_speed = wind_speed.raster.reproject_like(DEM, method="average")
            wind_speed.name = "sfcwind"
            self.set_forcing(wind_speed, name="climate/sfcwind")

        elif data_source == "cmip":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(f"Unknown data source: {data_source}")

        if calculate_SPEI:
            self.setup_SPEI()
        if calculate_GEV:
            self.setup_GEV()

    def download_ERA(
        self, variable, starttime: date, endtime: date, method: str, download_only=False
    ):
        # https://cds.climate.copernicus.eu/cdsapp#!/software/app-c3s-daily-era5-statistics?tab=appcode
        # https://earthscience.stackexchange.com/questions/24156/era5-single-level-calculate-relative-humidity
        import cdsapi

        """
        Download hourly ERA5 data for a specified time frame and bounding box.

        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

        """

        download_path = Path(self.root).parent / "preprocessing" / "climate" / "ERA5"
        download_path.mkdir(parents=True, exist_ok=True)

        def download(year):

            output_fn = download_path / f"{variable}_{year}.nc"
            if output_fn.exists():
                self.logger.info(f"ERA5 data already downloaded to {output_fn}")
            else:
                (xmin, ymin, xmax, ymax) = self.bounds

                # add buffer to bounding box. Resolution is 0.1 degrees, so add 0.1 degrees to each side
                xmin -= 0.1
                ymin -= 0.1
                xmax += 0.1
                ymax += 0.1

                c = cdsapi.Client()

                c.retrieve(
                    "reanalysis-era5-land",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": [
                            variable,
                        ],
                        "date": f"{year}-01-01/{year}-12-31",
                        "time": [
                            "00:00",
                            "01:00",
                            "02:00",
                            "03:00",
                            "04:00",
                            "05:00",
                            "06:00",
                            "07:00",
                            "08:00",
                            "09:00",
                            "10:00",
                            "11:00",
                            "12:00",
                            "13:00",
                            "14:00",
                            "15:00",
                            "16:00",
                            "17:00",
                            "18:00",
                            "19:00",
                            "20:00",
                            "21:00",
                            "22:00",
                            "23:00",
                        ],
                        "area": (ymax, xmin, ymin, xmax),  # North, West, South, East
                    },
                    output_fn,
                )
            return output_fn

        with concurrent.futures.ThreadPoolExecutor() as executor:
            years = range(starttime.year, endtime.year + 1)
            files = executor.map(download, years)

        if download_only:
            return

        ds = xr.open_dataset(output_fn)
        ds.raster.set_crs(4326)

        # assert there is only one data variable
        assert len(ds.data_vars) == 1

        # select the variable
        ds = ds[list(ds.data_vars)[0]].rename({"longitude": "x", "latitude": "y"})

        if method == "accumulation":
            # The accumulations in the short forecasts of ERA5-Land (with hourly steps from 01 to 24) are treated
            # the same as those in ERA-Interim or ERA-Interim/Land, i.e., they are accumulated from the beginning
            # of the forecast to the end of the forecast step. For example, runoff at day=D, step=12 will provide
            # runoff accumulated from day=D, time=0 to day=D, time=12. The maximum accumulation is over 24 hours,
            # i.e., from day=D, time=0 to day=D+1,time=0 (step=24).
            # forecasts are the difference between the current and previous time step
            hourly = ds.diff("time")
            # remove first from ds as well
            ds = ds.isel(time=slice(1, None))
            # except for UTC hour == 1, so assign the original data from ds to all values where the hour is 1
            hourly = hourly.where(hourly.time.dt.hour != 1, ds)
        elif method == "raw":
            hourly = ds
        else:
            raise NotImplementedError

        return hourly

    def snap_to_grid(self, ds, reference, relative_tollerance=0.02, ydim="y", xdim="x"):
        # make sure all datasets have more or less the same coordinates
        assert np.isclose(
            ds.coords[ydim].values,
            reference[ydim].values,
            atol=abs(ds.rio.resolution()[1] * relative_tollerance),
            rtol=0,
        ).all()
        assert np.isclose(
            ds.coords[xdim].values,
            reference[xdim].values,
            atol=abs(ds.rio.resolution()[0] * relative_tollerance),
            rtol=0,
        ).all()
        return ds.assign_coords({ydim: reference[ydim], xdim: reference[xdim]})

    def setup_1800arcsec_variables_isimip(
        self,
        forcing: str,
        variables: List[str],
        starttime: date,
        endtime: date,
        ssp: str,
    ):
        """
        Sets up the high-resolution climate variables for GEB.

        Parameters
        ----------
        variables : list of str
            The list of climate variables to set up.
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the high-resolution climate variables for GEB. It downloads the specified
        climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
        `download_isimip` method.

        The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
        then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

        The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable, forcing, ssp, starttime, endtime):
            self.logger.info(f"Setting up {variable}...")
            first_year_future_climate = 2015
            var = []
            if ssp == "picontrol":
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario=ssp,
                    variable=variable,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(
                    self.interpolate(
                        ds[variable].raster.clip_bbox(ds.raster.bounds),
                        "linear",
                        xdim="lon",
                        ydim="lat",
                    )
                )
            if (
                endtime.year < first_year_future_climate
                or starttime.year < first_year_future_climate
            ) and ssp != "picontrol":  # isimip cutoff date between historic and future climate
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario="historical",
                    variable=variable,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(
                    self.interpolate(
                        ds[variable].raster.clip_bbox(ds.raster.bounds),
                        "linear",
                        xdim="lon",
                        ydim="lat",
                    )
                )
            if (
                starttime.year >= first_year_future_climate
                or endtime.year >= first_year_future_climate
            ) and ssp != "picontrol":
                assert ssp is not None, "ssp must be specified for future climate"
                assert ssp != "historical", "historical scenarios run until 2014"
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario=ssp,
                    variable=variable,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(
                    self.interpolate(
                        ds[variable].raster.clip_bbox(ds.raster.bounds),
                        "linear",
                        xdim="lon",
                        ydim="lat",
                    )
                )

            var = xr.concat(var, dim="time")
            # assert that time is monotonically increasing with a constant step size
            assert (
                ds.time.diff("time").astype(np.int64)
                == (ds.time[1] - ds.time[0]).astype(np.int64)
            ).all(), "time is not monotonically increasing with a constant step size"
            var = var.rename({"lon": "x", "lat": "y"})
            self.logger.info(f"Completed {variable}")
            self.set_forcing(var, name=f"climate/{variable}")

        for variable in variables:
            download_variable(variable, forcing, ssp, starttime, endtime)

        # # Create a thread pool and map the set_forcing function to the variables
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(download_variable, variable, forcing, ssp, starttime, endtime) for variable in variables]

        # # Wait for all threads to complete
        # concurrent.futures.wait(futures)

    def setup_30arcsec_variables_isimip(
        self, variables: List[str], starttime: date, endtime: date
    ):
        """
        Sets up the high-resolution climate variables for GEB.

        Parameters
        ----------
        variables : list of str
            The list of climate variables to set up.
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the high-resolution climate variables for GEB. It downloads the specified
        climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
        `download_isimip` method.

        The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
        then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

        The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable, starttime, endtime):
            self.logger.info(f"Setting up {variable}...")
            ds = self.download_isimip(
                product="InputData",
                variable=variable,
                starttime=starttime,
                endtime=endtime,
                forcing="chelsa-w5e5v1.0",
                resolution="30arcsec",
            )
            ds = ds.rename({"lon": "x", "lat": "y"})
            var = ds[variable].raster.clip_bbox(ds.raster.bounds)
            var = self.snap_to_grid(var, self.grid)
            self.logger.info(f"Completed {variable}")
            self.set_forcing(var, name=f"climate/{variable}")

        for variable in variables:
            download_variable(variable, starttime, endtime)

        # # Create a thread pool and map the set_forcing function to the variables
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(download_variable, variable, starttime, endtime) for variable in variables]

        # # Wait for all threads to complete
        # concurrent.futures.wait(futures)

    def setup_hurs_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the relative humidity data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the relative humidity data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the relative humidity data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the relative humidity data for GEB. It first downloads the relative humidity
        data from the ISIMIP dataset for the specified time period using the `download_isimip` method. The data is downloaded
        at a 30 arcsec resolution.

        The method then downloads the monthly CHELSA-BIOCLIM+ relative humidity data at 30 arcsec resolution from the data
        catalog. The data is downloaded for each month in the specified time period and is clipped to the bounding box of
        the downloaded relative humidity data using the `clip_bbox` method of the `raster` object.

        The original ISIMIP data is then downscaled using the monthly CHELSA-BIOCLIM+ data. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting relative humidity data is set as forcing data in the model with names of the form 'climate/hurs'.
        """
        hurs_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        )  # some buffer to avoid edge effects / errors in ISIMIP API

        # just taking the years to simplify things
        start_year = starttime.year
        end_year = endtime.year

        chelsa_folder = (
            Path(self.root).parent
            / "preprocessing"
            / "climate"
            / "chelsa-bioclim+"
            / "hurs"
        )
        chelsa_folder.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Downloading/reading monthly CHELSA-BIOCLIM+ hurs data at 30 arcsec resolution"
        )
        hurs_ds_30sec, hurs_time = [], []
        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                fn = chelsa_folder / f"hurs_{year}_{month:02d}.nc"
                if not fn.exists():
                    hurs = self.data_catalog.get_rasterdataset(
                        f"CHELSA-BIOCLIM+_monthly_hurs_{month:02d}_{year}",
                        bbox=hurs_30_min.raster.bounds,
                        buffer=1,
                    )
                    del hurs.attrs["_FillValue"]
                    hurs.name = "hurs"
                    hurs.to_netcdf(fn)
                else:
                    hurs = xr.open_dataset(fn, chunks={"time": 1})["hurs"]
                # assert hasattr(hurs, "spatial_ref")
                hurs_ds_30sec.append(hurs)
                hurs_time.append(f"{year}-{month:02d}")

        hurs_ds_30sec = xr.concat(hurs_ds_30sec, dim="time").rename(
            {"x": "lon", "y": "lat"}
        )
        hurs_ds_30sec.rio.set_spatial_dims("lon", "lat", inplace=True)
        hurs_ds_30sec["time"] = pd.date_range(hurs_time[0], hurs_time[-1], freq="MS")

        hurs_output = xr.full_like(self.forcing["climate/tas"], np.nan)
        hurs_output.name = "hurs"
        hurs_output.attrs = {"units": "%", "long_name": "Relative humidity"}

        hurs_output = hurs_output.rename({"x": "lon", "y": "lat"}).rio.set_spatial_dims(
            "lon", "lat"
        )

        import xesmf as xe

        regridder = xe.Regridder(
            hurs_30_min.isel(time=0).drop_vars("time"),
            hurs_ds_30sec.isel(time=0).drop_vars("time"),
            "bilinear",
        )
        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                start_month = datetime(year, month, 1)
                end_month = datetime(year, month, monthrange(year, month)[1])

                w5e5_30min_sel = hurs_30_min.sel(time=slice(start_month, end_month))
                w5e5_regridded = regridder(w5e5_30min_sel) * 0.01  # convert to fraction
                assert (
                    w5e5_regridded >= 0.1
                ).all(), "too low values in relative humidity"
                assert (w5e5_regridded <= 1).all(), "relative humidity > 1"

                w5e5_regridded_mean = w5e5_regridded.mean(
                    dim="time"
                )  # get monthly mean
                w5e5_regridded_tr = np.log(
                    w5e5_regridded / (1 - w5e5_regridded)
                )  # assume beta distribuation => logit transform
                w5e5_regridded_mean_tr = np.log(
                    w5e5_regridded_mean / (1 - w5e5_regridded_mean)
                )  # logit transform

                chelsa = (
                    hurs_ds_30sec.sel(time=start_month) * 0.0001
                )  # convert to fraction
                assert (chelsa >= 0.1).all(), "too low values in relative humidity"
                assert (chelsa <= 1).all(), "relative humidity > 1"

                chelsa_tr = np.log(
                    chelsa / (1 - chelsa)
                )  # assume beta distribuation => logit transform

                difference = chelsa_tr - w5e5_regridded_mean_tr

                # apply difference to w5e5
                w5e5_regridded_tr_corr = w5e5_regridded_tr + difference
                w5e5_regridded_corr = (
                    1 / (1 + np.exp(-w5e5_regridded_tr_corr))
                ) * 100  # back transform
                w5e5_regridded_corr.raster.set_crs(4326)
                w5e5_regridded_corr_clipped = w5e5_regridded_corr[
                    "hurs"
                ].raster.clip_bbox(hurs_output.raster.bounds)
                w5e5_regridded_corr_clipped = (
                    w5e5_regridded_corr_clipped.rio.set_spatial_dims("lon", "lat")
                )

                hurs_output.loc[dict(time=slice(start_month, end_month))] = (
                    self.snap_to_grid(
                        w5e5_regridded_corr_clipped, hurs_output, xdim="lon", ydim="lat"
                    )
                )

        hurs_output = hurs_output.rename({"lon": "x", "lat": "y"})
        self.set_forcing(hurs_output, f"climate/hurs")

    def setup_longwave_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the longwave radiation data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the longwave radiation data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the longwave radiation data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the longwave radiation data for GEB. It first downloads the relative humidity,
        air temperature, and downward longwave radiation data from the ISIMIP dataset for the specified time period using the
        `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

        The method then regrids the downloaded data to the target grid using the `xe.Regridder` method. It calculates the
        saturation vapor pressure, water vapor pressure, clear-sky emissivity, all-sky emissivity, and cloud-based component
        of emissivity for the coarse and fine grids. It then downscales the longwave radiation data for the fine grid using
        the calculated all-sky emissivity and Stefan-Boltzmann constant. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting longwave radiation data is set as forcing data in the model with names of the form 'climate/rlds'.
        """
        x1 = 0.43
        x2 = 5.7
        sbc = 5.67e-8  # stefan boltzman constant [Js−1 m−2 K−4]

        es0 = 6.11  # reference saturation vapour pressure  [hPa]
        T0 = 273.15
        lv = 2.5e6  # latent heat of vaporization of water
        Rv = 461.5  # gas constant for water vapour [J K kg-1]

        target = self.forcing[f"climate/hurs"].rename({"x": "lon", "y": "lat"})

        hurs_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).hurs  # some buffer to avoid edge effects / errors in ISIMIP API
        tas_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="tas",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).tas  # some buffer to avoid edge effects / errors in ISIMIP API
        rlds_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="rlds",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).rlds  # some buffer to avoid edge effects / errors in ISIMIP API

        import xesmf as xe

        regridder = xe.Regridder(
            hurs_coarse.isel(time=0).drop_vars("time"), target, "bilinear"
        )

        hurs_coarse_regridded = regridder(hurs_coarse).rename({"lon": "x", "lat": "y"})
        tas_coarse_regridded = regridder(tas_coarse).rename({"lon": "x", "lat": "y"})
        rlds_coarse_regridded = regridder(rlds_coarse).rename({"lon": "x", "lat": "y"})

        hurs_fine = self.forcing[f"climate/hurs"]
        tas_fine = self.forcing[f"climate/tas"]

        # now ready for calculation:
        es_coarse = es0 * np.exp(
            (lv / Rv) * (1 / T0 - 1 / tas_coarse_regridded)
        )  # saturation vapor pressure
        pV_coarse = (
            hurs_coarse_regridded * es_coarse
        ) / 100  # water vapor pressure [hPa]

        es_fine = es0 * np.exp((lv / Rv) * (1 / T0 - 1 / tas_fine))
        pV_fine = (hurs_fine * es_fine) / 100  # water vapour pressure [hPa]

        e_cl_coarse = 0.23 + x1 * ((pV_coarse * 100) / tas_coarse_regridded) ** (1 / x2)
        # e_cl_coarse == clear-sky emissivity w5e5 (pV needs to be in Pa not hPa, hence *100)
        e_cl_fine = 0.23 + x1 * ((pV_fine * 100) / tas_fine) ** (1 / x2)
        # e_cl_fine == clear-sky emissivity target grid (pV needs to be in Pa not hPa, hence *100)

        e_as_coarse = rlds_coarse_regridded / (
            sbc * tas_coarse_regridded**4
        )  # all-sky emissivity w5e5
        e_as_coarse = xr.where(
            e_as_coarse < 1, e_as_coarse, 1
        )  # constrain all-sky emissivity to max 1
        assert (e_as_coarse <= 1).all(), "all-sky emissivity should be <= 1"
        delta_e = e_as_coarse - e_cl_coarse  # cloud-based component of emissivity w5e5

        e_as_fine = e_cl_fine + delta_e
        e_as_fine = xr.where(
            e_as_fine < 1, e_as_fine, 1
        )  # constrain all-sky emissivity to max 1
        assert (e_as_fine <= 1).all(), "all-sky emissivity should be <= 1"
        lw_fine = (
            e_as_fine * sbc * tas_fine**4
        )  # downscaled lwr! assume cloud e is the same

        lw_fine.name = "rlds"
        lw_fine = self.snap_to_grid(lw_fine, self.grid)
        self.set_forcing(lw_fine, name=f"climate/rlds")

    def setup_pressure_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the surface pressure data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the surface pressure data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the surface pressure data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the surface pressure data for GEB. It then downloads
        the orography data and surface pressure data from the ISIMIP dataset for the specified time period using the
        `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

        The method then regrids the orography and surface pressure data to the target grid using the `xe.Regridder` method.
        It corrects the surface pressure data for orography using the gravitational acceleration, molar mass of
        dry air, universal gas constant, and sea level standard temperature. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting surface pressure data is set as forcing data in the model with names of the form 'climate/ps'.
        """
        g = 9.80665  # gravitational acceleration [m/s2]
        M = 0.02896968  # molar mass of dry air [kg/mol]
        r0 = 8.314462618  # universal gas constant [J/(mol·K)]
        T0 = 288.16  # Sea level standard temperature  [K]

        target = self.forcing[f"climate/hurs"].rename({"x": "lon", "y": "lat"})
        pressure_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="psl",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).psl  # some buffer to avoid edge effects / errors in ISIMIP API

        orography = self.download_isimip(
            product="InputData", variable="orog", forcing="chelsa-w5e5v1.0", buffer=1
        ).orog  # some buffer to avoid edge effects / errors in ISIMIP API
        import xesmf as xe

        regridder = xe.Regridder(orography, target, "bilinear")
        orography = regridder(orography).rename({"lon": "x", "lat": "y"})

        regridder = xe.Regridder(
            pressure_30_min.isel(time=0).drop_vars("time"), target, "bilinear"
        )
        pressure_30_min_regridded = regridder(pressure_30_min).rename(
            {"lon": "x", "lat": "y"}
        )
        pressure_30_min_regridded_corr = pressure_30_min_regridded * np.exp(
            -(g * orography * M) / (T0 * r0)
        )

        pressure = xr.full_like(self.forcing[f"climate/hurs"], fill_value=np.nan)
        pressure.name = "ps"
        pressure.attrs = {"units": "Pa", "long_name": "surface pressure"}
        pressure.data = pressure_30_min_regridded_corr

        pressure = self.snap_to_grid(pressure, self.grid)
        self.set_forcing(pressure, name=f"climate/ps")

    def setup_wind_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the wind data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the wind data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the wind data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the wind data for GEB. It first downloads the global wind atlas data and
        regrids it to the target grid using the `xe.Regridder` method. It then downloads the 30-minute average wind data
        from the ISIMIP dataset for the specified time period and regrids it to the target grid using the `xe.Regridder`
        method.

        The method then creates a diff layer by assuming that wind follows a Weibull distribution and taking the log
        transform of the wind data. It then subtracts the log-transformed 30-minute average wind data from the
        log-transformed global wind atlas data to create the diff layer.

        The method then downloads the wind data from the ISIMIP dataset for the specified time period and regrids it to the
        target grid using the `xe.Regridder` method. It applies the diff layer to the log-transformed wind data and then
        exponentiates the result to obtain the corrected wind data. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting wind data is set as forcing data in the model with names of the form 'climate/wind'.
        """
        global_wind_atlas = self.data_catalog.get_rasterdataset(
            "global_wind_atlas", bbox=self.grid.raster.bounds, buffer=10
        ).rename({"x": "lon", "y": "lat"})
        target = self.grid["areamaps/grid_mask"].rename({"x": "lon", "y": "lat"})
        import xesmf as xe

        regridder = xe.Regridder(global_wind_atlas.copy(), target, "bilinear")
        global_wind_atlas_regridded = regridder(global_wind_atlas)

        wind_30_min_avg = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            starttime=date(2008, 1, 1),
            endtime=date(2017, 12, 31),
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind.mean(
            dim="time"
        )  # some buffer to avoid edge effects / errors in ISIMIP API
        regridder_30_min = xe.Regridder(wind_30_min_avg, target, "bilinear")
        wind_30_min_avg_regridded = regridder_30_min(wind_30_min_avg)

        # create diff layer:
        # assume wind follows weibull distribution => do log transform
        wind_30_min_avg_regridded_log = np.log(wind_30_min_avg_regridded)

        global_wind_atlas_regridded_log = np.log(global_wind_atlas_regridded)

        diff_layer = (
            global_wind_atlas_regridded_log - wind_30_min_avg_regridded_log
        )  # to be added to log-transformed daily

        wind_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30min_regridded = regridder_30_min(wind_30_min)
        wind_30min_regridded_log = np.log(wind_30min_regridded)

        wind_30min_regridded_log_corr = wind_30min_regridded_log + diff_layer
        wind_30min_regridded_corr = np.exp(wind_30min_regridded_log_corr)

        wind_output_clipped = wind_30min_regridded_corr.raster.clip_bbox(
            self.grid.raster.bounds
        )
        wind_output_clipped = wind_output_clipped.rename({"lon": "x", "lat": "y"})
        wind_output_clipped.name = "sfcwind"

        wind_output_clipped = self.snap_to_grid(wind_output_clipped, self.grid)
        self.set_forcing(wind_output_clipped, f"climate/sfcwind")

    def setup_SPEI(self):
        self.logger.info("setting up SPEI...")

        # assert input data have the same coordinates
        assert np.array_equal(
            self.forcing["climate/pr"].x, self.forcing["climate/tasmin"].x
        )
        assert np.array_equal(
            self.forcing["climate/pr"].x, self.forcing["climate/tasmax"].x
        )
        assert np.array_equal(
            self.forcing["climate/pr"].y, self.forcing["climate/tasmin"].y
        )
        assert np.array_equal(
            self.forcing["climate/pr"].y, self.forcing["climate/tasmax"].y
        )

        pet = xci.potential_evapotranspiration(
            tasmin=self.forcing["climate/tasmin"],
            tasmax=self.forcing["climate/tasmax"],
            method="BR65",
        )

        # Compute the potential evapotranspiration
        water_budget = xci._agro.water_budget(
            pr=self.forcing["climate/pr"], evspsblpot=pet
        )

        water_budget_positive = water_budget - 1.01 * water_budget.min()
        water_budget_positive.attrs = {"units": "kg m-2 s-1"}

        assert water_budget_positive.time.min().dt.date < date(
            2010, 1, 1
        ) and water_budget_positive.time.max().dt.date > date(
            1980, 1, 1
        ), "water budget data does not cover the reference period"
        wb_cal = water_budget_positive.sel(time=slice("1981-01-01", "2010-01-01"))
        assert wb_cal.time.size > 0

        # Compute the SPEI
        spei = xci._agro.standardized_precipitation_evapotranspiration_index(
            wb=water_budget_positive,
            wb_cal=wb_cal,
            freq="MS",
            window=12,
            dist="gamma",
            method="APP",
        )
        spei.attrs = {
            "units": "-",
            "long_name": "Standard Precipitation Evapotranspiration Index",
            "name": "spei",
        }
        spei.name = "spei"

        self.set_forcing(spei, name=f"climate/spei")

    def setup_GEV(self):
        self.logger.info("calculating GEV parameters...")

        # invert the values and take the max
        SPEI_changed = self.forcing[f"climate/spei"] * -1

        # Group the data by year and find the maximum monthly sum for each year
        SPEI_yearly_max = SPEI_changed.groupby("time.year").max(dim="time")
        SPEI_yearly_max = SPEI_yearly_max.rename({"year": "time"}).chunk({"time": -1})

        GEV = xci.stats.fit(SPEI_yearly_max, dist="genextreme").compute()
        GEV.name = "gev"

        self.set_grid(GEV.sel(dparams="c"), name=f"climate/gev_c")
        self.set_grid(GEV.sel(dparams="loc"), name=f"climate/gev_loc")
        self.set_grid(GEV.sel(dparams="scale"), name=f"climate/gev_scale")

    def setup_regions_and_land_use(
        self,
        region_database="gadm_level1",
        unique_region_id="UID",
        ISO3_column="ISO3",
        river_threshold=100,
    ):
        """
        Sets up the (administrative) regions and land use data for GEB. The regions can be used for multiple purposes,
        for example for creating the agents in the model, assigning unique crop prices and other economic variables
        per region and for aggregating the results.

        Parameters
        ----------
        region_database : str, optional
            The name of the region database to use. Default is 'gadm_level1'.
        unique_region_id : str, optional
            The name of the column in the region database that contains the unique region ID. Default is 'UID',
            which is the unique identifier for the GADM database.
        river_threshold : int, optional
            The threshold value to use when identifying rivers in the MERIT dataset. Default is 100.

        Notes
        -----
        This method sets up the regions and land use data for GEB. It first retrieves the region data from
        the specified region database and sets it as a geometry in the model. It then pads the subgrid to cover the entire
        region and retrieves the land use data from the ESA WorldCover dataset. The land use data is reprojected to the
        padded subgrid and the region ID is rasterized onto the subgrid. The cell area for each region is calculated and
        set as a grid in the model. The MERIT dataset is used to identify rivers, which are set as a grid in the model. The
        land use data is reclassified into five classes and set as a grid in the model. Finally, the cultivated land is
        identified and set as a grid in the model.

        The resulting grids are set as attributes of the model with names of the form 'areamaps/{grid_name}' or
        'landsurface/{grid_name}'.
        """
        self.logger.info(f"Preparing regions and land use data.")
        regions = self.data_catalog.get_geodataframe(
            region_database,
            geom=self.geoms["areamaps/region"],
            predicate="intersects",
        ).rename(columns={unique_region_id: "region_id", ISO3_column: "ISO3"})
        assert np.issubdtype(
            regions["region_id"].dtype, np.integer
        ), "Region ID must be integer"
        region_id_mapping = {
            i: region_id for region_id, i in enumerate(regions["region_id"])
        }
        regions["region_id"] = regions["region_id"].map(region_id_mapping)
        self.set_dict(region_id_mapping, name="areamaps/region_id_mapping")
        assert (
            "ISO3" in regions.columns
        ), f"Region database must contain ISO3 column ({self.data_catalog[region_database].path})"
        self.set_geoms(regions, name="areamaps/regions")

        region_bounds = self.geoms["areamaps/regions"].total_bounds

        resolution_x, resolution_y = self.subgrid[
            "areamaps/sub_grid_mask"
        ].rio.resolution()
        pad_minx = region_bounds[0] - abs(resolution_x) / 2.0
        pad_miny = region_bounds[1] - abs(resolution_y) / 2.0
        pad_maxx = region_bounds[2] + abs(resolution_x) / 2.0
        pad_maxy = region_bounds[3] + abs(resolution_y) / 2.0

        # TODO: Is there a better way to do this?
        padded_subgrid, region_subgrid_slice = pad_xy(
            self.subgrid["areamaps/sub_grid_mask"].rio,
            pad_minx,
            pad_miny,
            pad_maxx,
            pad_maxy,
            return_slice=True,
            constant_values=1,
        )
        padded_subgrid.raster.set_nodata(-1)
        self.set_region_subgrid(padded_subgrid, name="areamaps/region_mask")

        land_use = self.data_catalog.get_rasterdataset(
            "esa_worldcover_2020_v100",
            geom=self.geoms["areamaps/regions"],
            buffer=200,  # 2 km buffer
        )
        reprojected_land_use = land_use.raster.reproject_like(
            padded_subgrid, method="nearest"
        )

        region_raster = reprojected_land_use.raster.rasterize(
            self.geoms["areamaps/regions"],
            col_name="region_id",
            all_touched=True,
        )
        self.set_region_subgrid(region_raster, name="areamaps/region_subgrid")

        padded_cell_area = self.grid["areamaps/cell_area"].rio.pad_box(*region_bounds)
        # calculate the cell area for the grid for the entire region
        region_cell_area = calculate_cell_area(
            padded_cell_area.raster.transform, padded_cell_area.shape
        )

        # create subgrid for entire region
        region_cell_area_subgrid = hydromt.raster.full_from_transform(
            padded_cell_area.raster.transform * Affine.scale(1 / self.subgrid_factor),
            (
                padded_cell_area.raster.shape[0] * self.subgrid_factor,
                padded_cell_area.raster.shape[1] * self.subgrid_factor,
            ),
            nodata=np.nan,
            dtype=padded_cell_area.dtype,
            crs=padded_cell_area.raster.crs,
            name="areamaps/sub_grid_mask",
            lazy=True,
        )

        # calculate the cell area for the subgrid for the entire region
        region_cell_area_subgrid.data = (
            repeat_grid(region_cell_area, self.subgrid_factor) / self.subgrid_factor**2
        )

        # create new subgrid for the region without padding
        region_cell_area_subgrid_clipped_to_region = hydromt.raster.full(
            region_raster.raster.coords,
            nodata=np.nan,
            dtype=padded_cell_area.dtype,
            name="areamaps/sub_grid_mask",
            crs=region_raster.raster.crs,
            lazy=True,
        )

        # remove padding from region subgrid
        region_cell_area_subgrid_clipped_to_region.data = (
            region_cell_area_subgrid.raster.clip_bbox(
                (pad_minx, pad_miny, pad_maxx, pad_maxy)
            )
        )

        # set the cell area for the region subgrid
        self.set_region_subgrid(
            region_cell_area_subgrid_clipped_to_region,
            name="areamaps/region_cell_area_subgrid",
        )

        MERIT = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            variables=["upg"],
            bbox=padded_subgrid.rio.bounds(),
            buffer=300,  # 3 km buffer
            provider=self.data_provider,
        )
        # There is a half degree offset in MERIT data
        MERIT = MERIT.assign_coords(
            x=MERIT.coords["x"] + MERIT.rio.resolution()[0] / 2,
            y=MERIT.coords["y"] - MERIT.rio.resolution()[1] / 2,
        )

        # Assume all cells with at least x upstream cells are rivers.
        rivers = MERIT > river_threshold
        rivers = rivers.astype(np.int32)
        rivers.raster.set_nodata(-1)
        rivers = rivers.raster.reproject_like(reprojected_land_use, method="nearest")
        self.set_region_subgrid(rivers, name="landcover/rivers")

        hydro_land_use = reprojected_land_use.raster.reclassify(
            pd.DataFrame.from_dict(
                {
                    reprojected_land_use.raster.nodata: 5,  # no data, set to permanent water bodies because ocean
                    10: 0,  # tree cover
                    20: 1,  # shrubland
                    30: 1,  # grassland
                    40: 1,  # cropland, setting to non-irrigated. Initiated as irrigated based on agents
                    50: 4,  # built-up
                    60: 1,  # bare / sparse vegetation
                    70: 1,  # snow and ice
                    80: 5,  # permanent water bodies
                    90: 1,  # herbaceous wetland
                    95: 5,  # mangroves
                    100: 1,  # moss and lichen
                },
                orient="index",
                columns=["GEB_land_use_class"],
            ),
        )["GEB_land_use_class"]
        hydro_land_use = xr.where(
            rivers != 1, hydro_land_use, 5, keep_attrs=True
        )  # set rivers to 5 (permanent water bodies)
        hydro_land_use.raster.set_nodata(-1)

        self.set_region_subgrid(
            hydro_land_use, name="landsurface/full_region_land_use_classes"
        )

        cultivated_land = xr.where(
            (hydro_land_use == 1) & (reprojected_land_use == 40), 1, 0, keep_attrs=True
        )
        cultivated_land = cultivated_land.rio.set_nodata(-1)
        cultivated_land.rio.set_crs(reprojected_land_use.rio.crs)
        cultivated_land.rio.set_nodata(-1)

        self.set_region_subgrid(
            cultivated_land, name="landsurface/full_region_cultivated_land"
        )

        hydro_land_use_region = hydro_land_use.isel(region_subgrid_slice)

        # TODO: Doesn't work when using the original array. Somehow the dtype is changed on adding it to the subgrid. This is a workaround.
        self.set_subgrid(
            hydro_land_use_region.values, name="landsurface/land_use_classes"
        )

        cultivated_land_region = cultivated_land.isel(region_subgrid_slice)

        # Same workaround as above
        self.set_subgrid(
            cultivated_land_region.values, name="landsurface/cultivated_land"
        )

    def setup_economic_data(
        self, project_future_until_year=False, reference_start_year=2000
    ):
        """
        Sets up the economic data for GEB.

        Notes
        -----
        This method sets up the lending rates and inflation rates data for GEB. It first retrieves the
        lending rates and inflation rates data from the World Bank dataset using the `get_geodataframe` method of the
        `data_catalog` object. It then creates dictionaries to store the data for each region, with the years as the time
        dimension and the lending rates or inflation rates as the data dimension.

        The lending rates and inflation rates data are converted from percentage to rate by dividing by 100 and adding 1.
        The data is then stored in the dictionaries with the region ID as the key.

        The resulting lending rates and inflation rates data are set as forcing data in the model with names of the form
        'economics/lending_rates' and 'economics/inflation_rates', respectively.
        """
        self.logger.info("Setting up economic data")
        assert (
            not project_future_until_year
            or project_future_until_year > reference_start_year
        ), f"project_future_until_year ({project_future_until_year}) must be larger than reference_start_year ({reference_start_year})"

        lending_rates = self.data_catalog.get_dataframe("wb_lending_rate")
        inflation_rates = self.data_catalog.get_dataframe("wb_inflation_rate")

        lending_rates_dict, inflation_rates_dict = {"data": {}}, {"data": {}}
        years_lending_rates = [
            c
            for c in lending_rates.columns
            if c.isnumeric() and len(c) == 4 and int(c) >= 1900 and int(c) <= 3000
        ]
        lending_rates_dict["time"] = years_lending_rates

        years_inflation_rates = [
            c
            for c in inflation_rates.columns
            if c.isnumeric() and len(c) == 4 and int(c) >= 1900 and int(c) <= 3000
        ]
        inflation_rates_dict["time"] = years_inflation_rates

        for _, region in self.geoms["areamaps/regions"].iterrows():
            region_id = region["region_id"]
            ISO3 = region["ISO3"]

            lending_rates_country = (
                lending_rates.loc[
                    lending_rates["Country Code"] == ISO3, years_lending_rates
                ]
                / 100
                + 1
            )  # percentage to rate
            assert (
                len(lending_rates_country) == 1
            ), f"Expected one row for {ISO3}, got {len(lending_rates_country)}"
            lending_rates_dict["data"][region_id] = lending_rates_country.iloc[
                0
            ].tolist()

            inflation_rates_country = (
                inflation_rates.loc[
                    inflation_rates["Country Code"] == ISO3, years_inflation_rates
                ]
                / 100
                + 1
            )  # percentage to rate
            assert (
                len(inflation_rates_country) == 1
            ), f"Expected one row for {ISO3}, got {len(inflation_rates_country)}"
            inflation_rates_dict["data"][region_id] = inflation_rates_country.iloc[
                0
            ].tolist()

        if project_future_until_year:
            # convert to pandas dataframe
            inflation_rates = pd.DataFrame(
                inflation_rates_dict["data"], index=inflation_rates_dict["time"]
            ).dropna()
            lending_rates = pd.DataFrame(
                lending_rates_dict["data"], index=lending_rates_dict["time"]
            ).dropna()

            inflation_rates.index = inflation_rates.index.astype(int)
            # extend inflation rates to future
            mean_inflation_rate_since_reference_year = inflation_rates.loc[
                reference_start_year:
            ].mean(axis=0)
            inflation_rates = inflation_rates.reindex(
                range(inflation_rates.index.min(), project_future_until_year + 1)
            ).fillna(mean_inflation_rate_since_reference_year)

            inflation_rates_dict["time"] = inflation_rates.index.astype(str).tolist()
            inflation_rates_dict["data"] = inflation_rates.to_dict(orient="list")

            lending_rates.index = lending_rates.index.astype(int)
            # extend lending rates to future
            mean_lending_rate_since_reference_year = lending_rates.loc[
                reference_start_year:
            ].mean(axis=0)
            lending_rates = lending_rates.reindex(
                range(lending_rates.index.min(), project_future_until_year + 1)
            ).fillna(mean_lending_rate_since_reference_year)

            # convert back to dictionary
            lending_rates_dict["time"] = lending_rates.index.astype(str).tolist()
            lending_rates_dict["data"] = lending_rates.to_dict(orient="list")

        self.set_dict(inflation_rates_dict, name="economics/inflation_rates")
        self.set_dict(lending_rates_dict, name="economics/lending_rates")

    def setup_irrigation_sources(self, irrigation_sources):
        self.set_dict(irrigation_sources, name="agents/farmers/irrigation_sources")

    def setup_well_prices_by_reference_year(
        self,
        irrigation_maintenance: float,
        pump_cost: float,
        borewell_cost_1: float,
        borewell_cost_2: float,
        electricity_cost: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """
        Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        well_price : float
            The price of a well in the reference year.
        upkeep_price_per_m2 : float
            The upkeep price per square meter of a well in the reference year.
        reference_year : int
            The reference year for the well prices and upkeep prices.
        start_year : int
            The start year for the well prices and upkeep prices.
        end_year : int
            The end year for the well prices and upkeep prices.

        Notes
        -----
        This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
        retrieves the inflation rates data from the `economics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'economics/well_prices' and 'economics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["economics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "irrigation_maintenance": irrigation_maintenance,
            "pump_cost": pump_cost,
            "borewell_cost_1": borewell_cost_1,
            "borewell_cost_2": borewell_cost_2,
            "electricity_cost": electricity_cost,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for region in regions:
                prices = pd.Series(index=range(start_year, end_year + 1))
                prices.loc[reference_year] = initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year))
                        ]
                    )
                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict["data"][region] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_dict(prices_dict, name=f"economics/{price_type}")

    def setup_drip_irrigation_prices_by_reference_year(
        self,
        drip_irrigation_price: float,
        upkeep_price_per_m2: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """
        Sets up the drip_irrigation prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        drip_irrigation_price : float
            The price of a drip_irrigation in the reference year.
        upkeep_price_per_m2 : float
            The upkeep price per square meter of a drip_irrigation in the reference year.
        reference_year : int
            The reference year for the drip_irrigation prices and upkeep prices.
        start_year : int
            The start year for the drip_irrigation prices and upkeep prices.
        end_year : int
            The end year for the drip_irrigation prices and upkeep prices.

        Notes
        -----
        This method sets up the drip_irrigation prices and upkeep prices for the hydrological model based on a reference year. It first
        retrieves the inflation rates data from the `economics/inflation_rates` dictionary. It then creates dictionaries to
        store the drip_irrigation prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The drip_irrigation prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting drip_irrigation prices and upkeep prices data are set as dictionary with names of the form
        'economics/drip_irrigation_prices' and 'economics/upkeep_prices_drip_irrigation_per_m2', respectively.
        """
        self.logger.info("Setting up drip_irrigation prices by reference year")
        # create dictory with prices for drip_irrigation_prices per year by applying inflation rates
        inflation_rates = self.dict["economics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        drip_irrigation_prices_dict = {
            "time": list(range(start_year, end_year + 1)),
            "data": {},
        }
        for region in regions:
            drip_irrigation_prices = pd.Series(index=range(start_year, end_year + 1))
            drip_irrigation_prices.loc[reference_year] = drip_irrigation_price

            for year in range(reference_year + 1, end_year + 1):
                drip_irrigation_prices.loc[year] = (
                    drip_irrigation_prices[year - 1]
                    * inflation_rates["data"][region][
                        inflation_rates["time"].index(str(year))
                    ]
                )
            for year in range(reference_year - 1, start_year - 1, -1):
                drip_irrigation_prices.loc[year] = (
                    drip_irrigation_prices[year + 1]
                    / inflation_rates["data"][region][
                        inflation_rates["time"].index(str(year + 1))
                    ]
                )

            drip_irrigation_prices_dict["data"][
                region
            ] = drip_irrigation_prices.tolist()

        self.set_dict(
            drip_irrigation_prices_dict, name="economics/drip_irrigation_prices"
        )

        upkeep_prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}
        for region in regions:
            upkeep_prices = pd.Series(index=range(start_year, end_year + 1))
            upkeep_prices.loc[reference_year] = upkeep_price_per_m2

            for year in range(reference_year + 1, end_year + 1):
                upkeep_prices.loc[year] = (
                    upkeep_prices[year - 1]
                    * inflation_rates["data"][region][
                        inflation_rates["time"].index(str(year))
                    ]
                )
            for year in range(reference_year - 1, start_year - 1, -1):
                upkeep_prices.loc[year] = (
                    upkeep_prices[year + 1]
                    / inflation_rates["data"][region][
                        inflation_rates["time"].index(str(year + 1))
                    ]
                )

            upkeep_prices_dict["data"][region] = upkeep_prices.tolist()

        self.set_dict(
            upkeep_prices_dict, name="economics/upkeep_prices_drip_irrigation_per_m2"
        )

    def setup_farmers(self, farmers):
        """
        Sets up the farmers data for GEB.

        Parameters
        ----------
        farmers : pandas.DataFrame
            A DataFrame containing the farmer data.
        irrigation_sources : dict, optional
            A dictionary mapping irrigation source names to IDs.
        n_seasons : int, optional
            The number of seasons to simulate.

        Notes
        -----
        This method sets up the farmers data for GEB. It first retrieves the region data from the
        `areamaps/regions` and `areamaps/region_subgrid` grids. It then creates a `farms` grid with the same shape as the
        `region_subgrid` grid, with a value of -1 for each cell.

        For each region, the method clips the `cultivated_land` grid to the region and creates farms for the region using
        the `create_farms` function, using these farmlands as well as the dataframe of farmer agents. The resulting farms
        whose IDs correspondd to the IDs in the farmer dataframe are added to the `farms` grid for the region.

        The method then removes any farms that are outside the study area by using the `region_mask` grid. It then remaps
        the farmer IDs to a contiguous range of integers starting from 0.

        The resulting farms data is set as agents data in the model with names of the form 'agents/farmers/farms'. The
        crop names are mapped to IDs using the `crop_name_to_id` dictionary that was previously created. The resulting
        crop IDs are stored in the `season_#_crop` columns of the `farmers` DataFrame.

        If `irrigation_sources` is provided, the method sets the `irrigation_source` column of the `farmers` DataFrame to
        the corresponding IDs.

        Finally, the method sets the binary data for each column of the `farmers` DataFrame as agents data in the model
        with names of the form 'agents/farmers/{column}'.
        """
        regions = self.geoms["areamaps/regions"]
        regions_raster = self.region_subgrid["areamaps/region_subgrid"].compute()
        full_region_cultivated_land = self.region_subgrid[
            "landsurface/full_region_cultivated_land"
        ].compute()

        farms = hydromt.raster.full_like(regions_raster, nodata=-1, lazy=True)
        farms[:] = -1
        assert farms.min() >= -1  # -1 is nodata value, all farms should be positive

        for region_id in regions["region_id"]:
            self.logger.info(f"Creating farms for region {region_id}")
            region = regions_raster == region_id
            region_clip, bounds = clip_with_grid(region, region)

            cultivated_land_region = full_region_cultivated_land.isel(bounds)
            cultivated_land_region = xr.where(
                region_clip, cultivated_land_region, 0, keep_attrs=True
            )
            # TODO: Why does nodata value disappear?
            # self.dict['areamaps/region_id_mapping'][farmers['region_id']]
            farmer_region_ids = farmers["region_id"]
            farmers_region = farmers[farmer_region_ids == region_id]
            farms_region = create_farms(
                farmers_region, cultivated_land_region, farm_size_key="area_n_cells"
            )
            assert (
                farms_region.min() >= -1
            )  # -1 is nodata value, all farms should be positive
            farms[bounds] = xr.where(
                region_clip, farms_region, farms.isel(bounds), keep_attrs=True
            )
            farms = farms.compute()  # perhaps this helps with memory issues?

        farmers = farmers.drop("area_n_cells", axis=1)

        region_mask = self.region_subgrid["areamaps/region_mask"].astype(bool)

        # TODO: Again why is dtype changed? And export doesn't work?
        cut_farms = np.unique(
            xr.where(region_mask, farms.copy().values, -1, keep_attrs=True)
        )
        cut_farms = cut_farms[cut_farms != -1]

        assert farms.min() >= -1  # -1 is nodata value, all farms should be positive
        subgrid_farms = clip_with_grid(farms, ~region_mask)[0]

        subgrid_farms_in_study_area = xr.where(
            np.isin(subgrid_farms, cut_farms), -1, subgrid_farms, keep_attrs=True
        )
        farmers = farmers[~farmers.index.isin(cut_farms)]

        remap_farmer_ids = np.full(
            farmers.index.max() + 2, -1, dtype=np.int32
        )  # +1 because 0 is also a farm, +1 because no farm is -1, set to -1 in next step
        remap_farmer_ids[farmers.index] = np.arange(len(farmers))
        subgrid_farms_in_study_area = remap_farmer_ids[
            subgrid_farms_in_study_area.values
        ]

        farmers = farmers.reset_index(drop=True)

        assert np.setdiff1d(np.unique(subgrid_farms_in_study_area), -1).size == len(
            farmers
        )
        assert farmers.iloc[-1].name == subgrid_farms_in_study_area.max()

        self.set_subgrid(subgrid_farms_in_study_area, name="agents/farmers/farms")
        self.subgrid["agents/farmers/farms"].rio.set_nodata(-1)

        self.set_binary(farmers.index.values, name=f"agents/farmers/id")
        self.set_binary(farmers["region_id"].values, name=f"agents/farmers/region_id")

    def setup_farmers_from_csv(self, path=None):
        """
        Sets up the farmers data for GEB from a CSV file.

        Parameters
        ----------
        path : str
            The path to the CSV file containing the farmer data.

        Notes
        -----
        This method sets up the farmers data for GEB from a CSV file. It first reads the farmer data from
        the CSV file using the `pandas.read_csv` method.

        See the `setup_farmers` method for more information on how the farmer data is set up in the model.
        """
        if path is None:
            path = (
                Path(self.root).parent
                / "preprocessing"
                / "agents"
                / "farmers"
                / "farmers.csv"
            )
        farmers = pd.read_csv(path, index_col=0)
        self.setup_farmers(farmers)

    def setup_create_farms_simple(
        self,
        region_id_column="region_id",
        country_iso3_column="ISO3",
        farm_size_donor_countries=None,
        data_source="lowder",
        size_class_boundaries=None,
    ):
        """
        Sets up the farmers for GEB.

        Parameters
        ----------
        irrigation_sources : dict
            A dictionary of irrigation sources and their corresponding water availability in m^3/day.
        region_id_column : str, optional
            The name of the column in the region database that contains the region IDs. Default is 'UID'.
        country_iso3_column : str, optional
            The name of the column in the region database that contains the country ISO3 codes. Default is 'ISO3'.
        risk_aversion_mean : float, optional
            The mean of the normal distribution from which the risk aversion values are sampled. Default is 1.5.
        risk_aversion_standard_deviation : float, optional
            The standard deviation of the normal distribution from which the risk aversion values are sampled. Default is 0.5.

        Notes
        -----
        This method sets up the farmers for GEB. This is a simplified method that generates an example set of agent data.
        It first calculates the number of farmers and their farm sizes for each region based on the agricultural data for
        that region based on theamount of farm land and data from a global database on farm sizes per country. It then
        randomly assigns crops, irrigation sources, household sizes, and daily incomes and consumption levels to each farmer.

        A paper that reports risk aversion values for 75 countries is this one: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2646134
        """
        if data_source == "lowder":
            size_class_boundaries = {
                "< 1 Ha": (0, 10000),
                "1 - 2 Ha": (10000, 20000),
                "2 - 5 Ha": (20000, 50000),
                "5 - 10 Ha": (50000, 100000),
                "10 - 20 Ha": (100000, 200000),
                "20 - 50 Ha": (200000, 500000),
                "50 - 100 Ha": (500000, 1000000),
                "100 - 200 Ha": (1000000, 2000000),
                "200 - 500 Ha": (2000000, 5000000),
                "500 - 1000 Ha": (5000000, 10000000),
                "> 1000 Ha": (10000000, np.inf),
            }
        else:
            assert size_class_boundaries is not None
            assert (
                farm_size_donor_countries is None
            ), "farm_size_donor_countries is only used for lowder data"

        cultivated_land = self.region_subgrid[
            "landsurface/full_region_cultivated_land"
        ].compute()
        regions_grid = self.region_subgrid["areamaps/region_subgrid"].compute()
        cell_area = self.region_subgrid["areamaps/region_cell_area_subgrid"].compute()

        regions_shapes = self.geoms["areamaps/regions"]
        if data_source == "lowder":
            assert (
                country_iso3_column in regions_shapes.columns
            ), f"Region database must contain {country_iso3_column} column ({self.data_catalog['gadm_level1'].path})"

            farm_sizes_per_region = (
                self.data_catalog.get_dataframe("lowder_farm_sizes")
                .dropna(subset=["Total"], axis=0)
                .drop(["empty", "income class"], axis=1)
            )
            farm_sizes_per_region["Country"] = farm_sizes_per_region["Country"].ffill()
            # Remove preceding and trailing white space from country names
            farm_sizes_per_region["Country"] = farm_sizes_per_region[
                "Country"
            ].str.strip()
            farm_sizes_per_region["Census Year"] = farm_sizes_per_region[
                "Country"
            ].ffill()

            # convert country names to ISO3 codes
            iso3_codes = {
                "Albania": "ALB",
                "Algeria": "DZA",
                "American Samoa": "ASM",
                "Argentina": "ARG",
                "Austria": "AUT",
                "Bahamas": "BHS",
                "Barbados": "BRB",
                "Belgium": "BEL",
                "Brazil": "BRA",
                "Bulgaria": "BGR",
                "Burkina Faso": "BFA",
                "Chile": "CHL",
                "Colombia": "COL",
                "Côte d'Ivoire": "CIV",
                "Croatia": "HRV",
                "Cyprus": "CYP",
                "Czech Republic": "CZE",
                "Democratic Republic of the Congo": "COD",
                "Denmark": "DNK",
                "Dominica": "DMA",
                "Ecuador": "ECU",
                "Egypt": "EGY",
                "Estonia": "EST",
                "Ethiopia": "ETH",
                "Fiji": "FJI",
                "Finland": "FIN",
                "France": "FRA",
                "French Polynesia": "PYF",
                "Georgia": "GEO",
                "Germany": "DEU",
                "Greece": "GRC",
                "Grenada": "GRD",
                "Guam": "GUM",
                "Guatemala": "GTM",
                "Guinea": "GIN",
                "Honduras": "HND",
                "India": "IND",
                "Indonesia": "IDN",
                "Iran (Islamic Republic of)": "IRN",
                "Ireland": "IRL",
                "Italy": "ITA",
                "Japan": "JPN",
                "Jamaica": "JAM",
                "Jordan": "JOR",
                "Korea, Rep. of": "KOR",
                "Kyrgyzstan": "KGZ",
                "Lao People's Democratic Republic": "LAO",
                "Latvia": "LVA",
                "Lebanon": "LBN",
                "Lithuania": "LTU",
                "Luxembourg": "LUX",
                "Malta": "MLT",
                "Morocco": "MAR",
                "Myanmar": "MMR",
                "Namibia": "NAM",
                "Nepal": "NPL",
                "Netherlands": "NLD",
                "Nicaragua": "NIC",
                "Northern Mariana Islands": "MNP",
                "Norway": "NOR",
                "Pakistan": "PAK",
                "Panama": "PAN",
                "Paraguay": "PRY",
                "Peru": "PER",
                "Philippines": "PHL",
                "Poland": "POL",
                "Portugal": "PRT",
                "Puerto Rico": "PRI",
                "Qatar": "QAT",
                "Romania": "ROU",
                "Saint Lucia": "LCA",
                "Saint Vincent and the Grenadines": "VCT",
                "Samoa": "WSM",
                "Senegal": "SEN",
                "Serbia": "SRB",
                "Sweden": "SWE",
                "Switzerland": "CHE",
                "Thailand": "THA",
                "Trinidad and Tobago": "TTO",
                "Turkey": "TUR",
                "Uganda": "UGA",
                "United Kingdom": "GBR",
                "United States of America": "USA",
                "Uruguay": "URY",
                "Venezuela (Bolivarian Republic of)": "VEN",
                "Virgin Islands, United States": "VIR",
                "Yemen": "YEM",
                "Cook Islands": "COK",
                "French Guiana": "GUF",
                "Guadeloupe": "GLP",
                "Martinique": "MTQ",
                "Réunion": "REU",
                "Canada": "CAN",
                "China": "CHN",
                "Guinea Bissau": "GNB",
                "Hungary": "HUN",
                "Lesotho": "LSO",
                "Libya": "LBY",
                "Malawi": "MWI",
                "Mozambique": "MOZ",
                "New Zealand": "NZL",
                "Slovakia": "SVK",
                "Slovenia": "SVN",
                "Spain": "ESP",
                "St. Kitts & Nevis": "KNA",
                "Viet Nam": "VNM",
                "Australia": "AUS",
                "Djibouti": "DJI",
                "Mali": "MLI",
                "Togo": "TGO",
                "Zambia": "ZMB",
            }
            farm_sizes_per_region["ISO3"] = farm_sizes_per_region["Country"].map(
                iso3_codes
            )
            assert (
                not farm_sizes_per_region["ISO3"].isna().any()
            ), f"Found {farm_sizes_per_region['ISO3'].isna().sum()} countries without ISO3 code"
        else:
            # load data source
            farm_sizes_per_region = pd.read_excel(
                data_source["farm_size"], index_col=(0, 1, 2)
            )
            n_farms_per_region = pd.read_excel(
                data_source["n_farms"],
                index_col=(0, 1, 2),
            )

        all_agents = []
        self.logger.debug(f"Starting processing of {len(regions_shapes)} regions")
        for _, region in regions_shapes.iterrows():
            UID = region[region_id_column]
            if data_source == "lowder":
                country_ISO3 = region[country_iso3_column]
                if farm_size_donor_countries:
                    country_ISO3 = farm_size_donor_countries.get(
                        country_ISO3, country_ISO3
                    )
            else:
                state, district, tehsil = (
                    region["state_name"],
                    region["district_n"],
                    region["sub_dist_1"],
                )

            self.logger.debug(f"Processing region {UID}")

            cultivated_land_region_total_cells = (
                ((regions_grid == UID) & (cultivated_land == True)).sum().compute()
            )
            total_cultivated_land_area_lu = (
                (((regions_grid == UID) & (cultivated_land == True)) * cell_area)
                .sum()
                .compute()
            )
            if (
                total_cultivated_land_area_lu == 0
            ):  # when no agricultural area, just continue as there will be no farmers. Also avoiding some division by 0 errors.
                continue

            average_cell_area_region = (
                cell_area.where(((regions_grid == UID) & (cultivated_land == True)))
                .mean()
                .compute()
            )

            if data_source == "lowder":
                region_farm_sizes = farm_sizes_per_region.loc[
                    (farm_sizes_per_region["ISO3"] == country_ISO3)
                ].drop(["Country", "Census Year", "Total"], axis=1)
                assert (
                    len(region_farm_sizes) == 2
                ), f"Found {len(region_farm_sizes) / 2} region_farm_sizes for {country_ISO3}"

                region_n_holdings = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"] == "Holdings"
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", "0")
                    .astype(np.int64)
                )
                agricultural_area_db_ha = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"]
                        == "Agricultural area (Ha) "
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", "0")
                    .astype(np.int64)
                )
                agricultural_area_db = agricultural_area_db_ha * 10000
                region_farm_sizes = agricultural_area_db / region_n_holdings
            else:
                region_farm_sizes = farm_sizes_per_region.loc[(state, district, tehsil)]
                region_n_holdings = n_farms_per_region.loc[(state, district, tehsil)]
                agricultural_area_db = region_farm_sizes * region_n_holdings

            total_cultivated_land_area_db = agricultural_area_db.sum()

            n_cells_per_size_class = pd.Series(0, index=region_n_holdings.index)

            for size_class in agricultural_area_db.index:
                if (
                    region_n_holdings[size_class] > 0
                ):  # if no holdings, no need to calculate
                    region_n_holdings[size_class] = region_n_holdings[size_class] * (
                        total_cultivated_land_area_lu / total_cultivated_land_area_db
                    )
                    n_cells_per_size_class.loc[size_class] = (
                        region_n_holdings[size_class]
                        * region_farm_sizes[size_class]
                        / average_cell_area_region
                    )
                    assert not np.isnan(n_cells_per_size_class.loc[size_class])

            assert math.isclose(
                cultivated_land_region_total_cells, n_cells_per_size_class.sum()
            )

            whole_cells_per_size_class = (n_cells_per_size_class // 1).astype(int)
            leftover_cells_per_size_class = n_cells_per_size_class % 1
            whole_cells = whole_cells_per_size_class.sum()
            n_missing_cells = cultivated_land_region_total_cells - whole_cells
            assert n_missing_cells <= len(agricultural_area_db)

            index = list(
                zip(
                    leftover_cells_per_size_class.index,
                    leftover_cells_per_size_class % 1,
                )
            )
            n_cells_to_add = sorted(index, key=lambda x: x[1], reverse=True)[
                : n_missing_cells.compute().item()
            ]
            whole_cells_per_size_class.loc[[p[0] for p in n_cells_to_add]] += 1

            region_agents = []
            for size_class in whole_cells_per_size_class.index:
                # if no cells for this size class, just continue
                if whole_cells_per_size_class.loc[size_class] == 0:
                    continue

                number_of_agents_size_class = round(
                    region_n_holdings[size_class].compute().item()
                )
                # if there is agricultural land, but there are no agents rounded down, we assume there is one agent
                if (
                    number_of_agents_size_class == 0
                    and whole_cells_per_size_class[size_class] > 0
                ):
                    number_of_agents_size_class = 1

                min_size_m2, max_size_m2 = size_class_boundaries[size_class]
                if max_size_m2 in (np.inf, "inf", "infinity", "Infinity"):
                    max_size_m2 = region_farm_sizes[size_class] * 2

                min_size_cells = int(min_size_m2 / average_cell_area_region)
                min_size_cells = max(
                    min_size_cells, 1
                )  # farm can never be smaller than one cell
                max_size_cells = (
                    int(max_size_m2 / average_cell_area_region) - 1
                )  # otherwise they overlap with next size class
                mean_cells_per_agent = int(
                    region_farm_sizes[size_class] / average_cell_area_region
                )

                if (
                    mean_cells_per_agent < min_size_cells
                    or mean_cells_per_agent > max_size_cells
                ):  # there must be an error in the data, thus assume centred
                    mean_cells_per_agent = (min_size_cells + max_size_cells) // 2

                population = pd.DataFrame(index=range(number_of_agents_size_class))

                offset = (
                    whole_cells_per_size_class[size_class]
                    - number_of_agents_size_class * mean_cells_per_agent
                )

                if (
                    number_of_agents_size_class * mean_cells_per_agent + offset
                    < min_size_cells * number_of_agents_size_class
                ):
                    min_size_cells = (
                        number_of_agents_size_class * mean_cells_per_agent + offset
                    ) // number_of_agents_size_class
                if (
                    number_of_agents_size_class * mean_cells_per_agent + offset
                    > max_size_cells * number_of_agents_size_class
                ):
                    max_size_cells = (
                        number_of_agents_size_class * mean_cells_per_agent + offset
                    ) // number_of_agents_size_class + 1

                n_farms_size_class, farm_sizes_size_class = get_farm_distribution(
                    number_of_agents_size_class,
                    min_size_cells,
                    max_size_cells,
                    mean_cells_per_agent,
                    offset,
                    self.logger,
                )
                assert n_farms_size_class.sum() == number_of_agents_size_class
                assert (farm_sizes_size_class > 0).all()
                assert (
                    n_farms_size_class * farm_sizes_size_class
                ).sum() == whole_cells_per_size_class[size_class]
                farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)
                np.random.shuffle(farm_sizes)
                population["area_n_cells"] = farm_sizes
                region_agents.append(population)

                assert (
                    population["area_n_cells"].sum()
                    == whole_cells_per_size_class[size_class]
                )

            region_agents = pd.concat(region_agents, ignore_index=True)
            region_agents["region_id"] = UID
            all_agents.append(region_agents)

        farmers = pd.concat(all_agents, ignore_index=True)
        self.setup_farmers(farmers)

    def setup_farmer_characteristics_simple(
        self,
        irrigation_sources=None,
        irrigation_choice=0,
        crop_choices=None,
        risk_aversion_mean=1.5,
        risk_aversion_standard_deviation=0.5,
        interest_rate=0.05,
        discount_rate=0.1,
        n_seasons=3,
    ):
        n_farmers = self.binary["agents/farmers/id"].size

        for season in range(1, n_seasons + 1):
            # randomly sample from crops
            if crop_choices[season - 1] == "random":
                crop_ids = [int(ID) for ID in self.dict["crops/crop_ids"].keys()]
                farmer_crops = random.choices(crop_ids, k=n_farmers)
            else:
                farmer_crops = np.full(
                    n_farmers, crop_choices[season - 1], dtype=np.int32
                )
            self.set_binary(farmer_crops, name=f"agents/farmers/season_#{season}_crop")

        if irrigation_choice == "random":
            # randomly sample from irrigation sources
            irrigation_source = random.choices(
                list(irrigation_sources.values()), k=n_farmers
            )
        else:
            irrigation_source = np.full(n_farmers, irrigation_choice, dtype=np.int32)
        self.set_binary(irrigation_source, name="agents/farmers/irrigation_source")

        household_size = random.choices([1, 2, 3, 4, 5, 6, 7], k=n_farmers)
        self.set_binary(household_size, name="agents/farmers/household_size")

        daily_non_farm_income_family = random.choices([50, 100, 200, 500], k=n_farmers)
        self.set_binary(
            daily_non_farm_income_family,
            name="agents/farmers/daily_non_farm_income_family",
        )

        daily_consumption_per_capita = random.choices([50, 100, 200, 500], k=n_farmers)
        self.set_binary(
            daily_consumption_per_capita,
            name="agents/farmers/daily_consumption_per_capita",
        )

        risk_aversion = np.random.normal(
            loc=risk_aversion_mean,
            scale=risk_aversion_standard_deviation,
            size=n_farmers,
        )
        self.set_binary(risk_aversion, name="agents/farmers/risk_aversion")

        interest_rate = np.full(n_farmers, interest_rate, dtype=np.float32)
        self.set_binary(interest_rate, name="agents/farmers/interest_rate")

        discount_rate = np.full(n_farmers, discount_rate, dtype=np.float32)
        self.set_binary(discount_rate, name="agents/farmers/discount_rate")

    def setup_farmer_characteristics_india(
        self,
        n_seasons,
        crop_choices,
        risk_aversion_mean,
        risk_aversion_standard_deviation,
        discount_rate,
        interest_rate,
        well_irrigated_ratio,
    ):
        n_farmers = self.binary["agents/farmers/id"].size

        for season in range(1, n_seasons + 1):
            # randomly sample from crops
            if crop_choices[season - 1] == "random":
                crop_ids = [int(ID) for ID in self.dict["crops/crop_ids"].keys()]
                farmer_crops = random.choices(crop_ids, k=n_farmers)
            else:
                farmer_crops = np.full(
                    n_farmers, crop_choices[season - 1], dtype=np.int32
                )
            self.set_binary(farmer_crops, name=f"agents/farmers/season_#{season}_crop")

        irrigation_sources = self.dict["agents/farmers/irrigation_sources"]

        irrigation_source = np.full(n_farmers, irrigation_sources["no"], dtype=np.int32)

        farms = self.subgrid["agents/farmers/farms"]
        if "routing/lakesreservoirs/subcommand_areas" in self.subgrid:
            command_areas = self.subgrid["routing/lakesreservoirs/subcommand_areas"]
            canal_irrigated_farms = np.unique(farms.where(command_areas != -1, -1))
            canal_irrigated_farms = canal_irrigated_farms[canal_irrigated_farms != -1]
            irrigation_source[canal_irrigated_farms] = irrigation_sources["canal"]

        well_irrigated_farms = np.random.choice(
            [0, 1],
            size=n_farmers,
            replace=True,
            p=[1 - well_irrigated_ratio, well_irrigated_ratio],
        ).astype(bool)
        irrigation_source[
            (well_irrigated_farms) & (irrigation_source == irrigation_sources["no"])
        ] = irrigation_sources["well"]

        self.set_binary(irrigation_source, name="agents/farmers/irrigation_source")

        # get farmer locations
        vertical_index = (
            np.arange(farms.shape[0])
            .repeat(farms.shape[1])
            .reshape(farms.shape)[farms != -1]
        )
        horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(
            farms.shape
        )[farms != -1]
        farms_flattened = farms.values[farms.values != -1]
        pixels = np.zeros((n_farmers, 2), dtype=np.int32)
        pixels[:, 0] = np.round(
            np.bincount(farms_flattened, horizontal_index)
            / np.bincount(farms_flattened)
        ).astype(int)
        pixels[:, 1] = np.round(
            np.bincount(farms_flattened, vertical_index) / np.bincount(farms_flattened)
        ).astype(int)

        locations = pixels_to_coords(pixels + 0.5, farms.raster.transform.to_gdal())
        locations = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(locations[:, 0], locations[:, 1]),
            crs="EPSG:4326",
        )  # convert locations to geodataframe

        # GLOPOP-S uses the GDL regions. So we need to get the GDL region for each farmer using their location
        GDL_regions = self.data_catalog.get_geodataframe(
            "GDL_regions_v4", geom=self.geoms["areamaps/region"], variables=["GDLcode"]
        )
        GDL_region_per_farmer = gpd.sjoin(
            locations, GDL_regions, how="left", op="within"
        )

        # ensure that each farmer has a region
        assert GDL_region_per_farmer["GDLcode"].notna().all()

        # Load GLOPOP-S data. This is a binary file and has no proper loading in hydromt. So we use the data catalog to get the path and format the path with the regions and load it with NumPy
        GLOPOP_S = self.data_catalog.get_source("GLOPOP-S")

        GLOPOP_S_attribute_names = [
            "economic_class",
            "settlement_type_rural",
            "farmer",
            "age",
            "gender",
            "education_level",
            "household_type",
            "household_ID",
            "relation_to_household_head",
            "household_size_category",
        ]
        # Get list of unique GDL codes from farmer dataframe
        GDL_region_per_farmer["household_size"] = np.full(
            len(GDL_region_per_farmer), -1, dtype=np.int32
        )
        for GDL_region, farmers_GDL_region in GDL_region_per_farmer.groupby("GDLcode"):
            GLOPOP_S_region = np.fromfile(
                GLOPOP_S.path.format(region=GDL_region),
                dtype=np.int32,
            )

            n_people = GLOPOP_S_region.size // len(GLOPOP_S_attribute_names)
            GLOPOP_S_region = pd.DataFrame(
                np.reshape(
                    GLOPOP_S_region, (len(GLOPOP_S_attribute_names), n_people)
                ).transpose(),
                columns=GLOPOP_S_attribute_names,
            ).drop(
                ["economic_class", "settlement_type_rural", "household_size_category"],
                axis=1,
            )
            # select farmers only
            GLOPOP_S_region = GLOPOP_S_region[GLOPOP_S_region["farmer"] == 1].drop(
                "farmer", axis=1
            )

            # Select a random sample of farmers from the database
            GLOPOP_S_household_IDs = GLOPOP_S_region["household_ID"].unique()
            if GLOPOP_S_household_IDs.size > len(farmers_GDL_region):
                GLOPOP_S_household_IDs = np.random.choice(
                    GLOPOP_S_household_IDs, size=len(farmers_GDL_region), replace=False
                )
                GLOPOP_S_region = GLOPOP_S_region[
                    GLOPOP_S_region["household_ID"].isin(GLOPOP_S_household_IDs)
                ]
            else:
                # TODO: Implement upsampling of GLOPOP_S data
                raise NotImplementedError

            households_region = GLOPOP_S_region.groupby("household_ID")
            household_sizes_region = households_region.size().values.astype(np.int32)
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "household_size"] = (
                household_sizes_region
            )

        # assert non of the household sizes are placeholder value -1
        assert (GDL_region_per_farmer["household_size"] != -1).all()

        self.set_binary(
            GDL_region_per_farmer["household_size"].values,
            name="agents/farmers/household_size",
        )

        daily_non_farm_income_family = random.choices([50, 100, 200, 500], k=n_farmers)
        self.set_binary(
            daily_non_farm_income_family,
            name="agents/farmers/daily_non_farm_income_family",
        )

        daily_consumption_per_capita = random.choices([50, 100, 200, 500], k=n_farmers)
        self.set_binary(
            daily_consumption_per_capita,
            name="agents/farmers/daily_consumption_per_capita",
        )

        risk_aversion = np.random.normal(
            loc=risk_aversion_mean,
            scale=risk_aversion_standard_deviation,
            size=n_farmers,
        )
        self.set_binary(risk_aversion, name="agents/farmers/risk_aversion")

        interest_rate = np.full(n_farmers, interest_rate, dtype=np.float32)
        self.set_binary(interest_rate, name="agents/farmers/interest_rate")

        discount_rate = np.full(n_farmers, discount_rate, dtype=np.float32)
        self.set_binary(discount_rate, name="agents/farmers/discount_rate")

    def setup_population(self):
        populaton_map = self.data_catalog.get_rasterdataset(
            "ghs_pop_2020_54009_v2023a", bbox=self.bounds
        )
        populaton_map_values = np.round(populaton_map.values).astype(np.int32)
        populaton_map_values[populaton_map_values < 0] = 0  # -200 is nodata value

        locations, sizes = generate_locations(
            population=populaton_map_values,
            geotransform=populaton_map.raster.transform.to_gdal(),
            mean_household_size=5,
        )

        transformer = pyproj.Transformer.from_crs(
            populaton_map.raster.crs, self.epsg, always_xy=True
        )
        locations[:, 0], locations[:, 1] = transformer.transform(
            locations[:, 0], locations[:, 1]
        )

        # sample_locatons = locations[::10]
        # import matplotlib.pyplot as plt
        # from scipy.stats import gaussian_kde

        # xy = np.vstack([sample_locatons[:, 0], sample_locatons[:, 1]])
        # z = gaussian_kde(xy)(xy)
        # plt.scatter(sample_locatons[:, 0], sample_locatons[:, 1], c=z, s=100)
        # plt.savefig('population.png')

        self.set_binary(sizes, name="agents/households/sizes")
        self.set_binary(locations, name="agents/households/locations")

        return None

    def setup_assets(self, feature_types):
        import osm_flex.download
        import osm_flex.extract

        if isinstance(feature_types, str):
            feature_types = [feature_types]

        OSM_data_dir = Path(self.root).parent / "preprocessing" / "osm"
        OSM_data_dir.mkdir(exist_ok=True, parents=True)

        index_file = OSM_data_dir / "geofabrik_region_index.geojson"
        fetch_and_save(
            "https://download.geofabrik.de/index-v1.json", index_file, overwrite=False
        )

        index = gpd.read_file(index_file)
        # remove Dach region as all individual regions within dach countries are also in the index
        index = index[index["id"] != "dach"]

        # find all regions that intersect with the bbox
        intersecting_regions = index[index.intersects(self.region.geometry[0])]

        def filter_regions(ID, parents):
            return ID not in parents

        intersecting_regions = intersecting_regions[
            intersecting_regions["id"].apply(
                lambda x: filter_regions(x, intersecting_regions["parent"].tolist())
            )
        ]

        # download all regions
        all_features = {}
        for _, row in tqdm(intersecting_regions.iterrows()):
            url = json.loads(row["urls"])["pbf"]
            filepath = OSM_data_dir / url.split("/")[-1]
            fetch_and_save(url, filepath, overwrite=False)
            for feature_type in feature_types:
                if feature_type not in all_features:
                    all_features[feature_type] = []
                features = osm_flex.extract.extract_cis(filepath, feature_type)
                # features = features.clip(self.geoms["areamaps/region"])
                features = gpd.sjoin(
                    features,
                    self.geoms["areamaps/region"],
                    how="inner",
                    op="intersects",
                )
                all_features[feature_type].append(features)

        for feature_type in feature_types:
            features = pd.concat(all_features[feature_type], ignore_index=True)
            self.set_geoms(features, name=f"assets/{feature_type}")

    def interpolate(self, ds, interpolation_method, ydim="y", xdim="x"):
        out_ds = ds.interp(
            method=interpolation_method,
            **{
                ydim: self.grid.y.rename({"y": ydim}),
                xdim: self.grid.x.rename({"x": xdim}),
            },
        )
        assert len(ds.dims) == len(out_ds.dims)
        return out_ds

    def download_isimip(
        self,
        product,
        variable,
        forcing,
        starttime=None,
        endtime=None,
        simulation_round="ISIMIP3a",
        climate_scenario="obsclim",
        resolution=None,
        buffer=0,
    ):
        """
        Downloads ISIMIP climate data for GEB.

        Parameters
        ----------
        product : str
            The name of the ISIMIP product to download.
        variable : str
            The name of the climate variable to download.
        forcing : str
            The name of the climate forcing to download.
        starttime : date, optional
            The start date of the data. Default is None.
        endtime : date, optional
            The end date of the data. Default is None.
        resolution : str, optional
            The resolution of the data to download. Default is None.
        buffer : int, optional
            The buffer size in degrees to add to the bounding box of the data to download. Default is 0.

        Returns
        -------
        xr.Dataset
            The downloaded climate data as an xarray dataset.

        Notes
        -----
        This method downloads ISIMIP climate data for GEB. It first retrieves the dataset
        metadata from the ISIMIP repository using the specified `product`, `variable`, `forcing`, and `resolution`
        parameters. It then downloads the data files that match the specified `starttime` and `endtime` parameters, and
        extracts them to the specified `download_path` directory.

        The resulting climate data is returned as an xarray dataset. The dataset is assigned the coordinate reference system
        EPSG:4326, and the spatial dimensions are set to 'lon' and 'lat'.
        """
        # if starttime is specified, endtime must be specified as well
        assert (starttime is None) == (endtime is None)

        client = ISIMIPClient()
        download_path = (
            Path(self.root).parent / "preprocessing" / "climate" / forcing / variable
        )
        download_path.mkdir(parents=True, exist_ok=True)

        # Code to get data from disk rather than server.
        parse_files = []
        for file in os.listdir(download_path):
            if file.endswith(".nc"):
                fp = download_path / file
                parse_files.append(fp)

        # get the dataset metadata from the ISIMIP repository
        response = client.datasets(
            simulation_round=simulation_round,
            product=product,
            climate_forcing=forcing,
            climate_scenario=climate_scenario,
            climate_variable=variable,
            resolution=resolution,
        )
        assert len(response["results"]) == 1
        dataset = response["results"][0]
        files = dataset["files"]

        xmin, ymin, xmax, ymax = self.bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer

        if variable == "orog":
            assert len(files) == 1
            filename = files[0][
                "name"
            ]  # global should be included due to error in ISIMIP API .replace('_global', '')
            parse_files = [filename]
            if not (download_path / filename).exists():
                download_files = [files[0]["path"]]
            else:
                download_files = []

        else:
            assert starttime is not None and endtime is not None
            download_files = []
            parse_files = []
            for file in files:
                name = file["name"]
                assert name.endswith(".nc")
                splitted_filename = name.split("_")
                date = splitted_filename[-1].split(".")[0]
                if "-" in date:
                    start_date, end_date = date.split("-")
                    start_date = datetime.strptime(start_date, "%Y%m%d").date()
                    end_date = datetime.strptime(end_date, "%Y%m%d").date()
                elif len(date) == 6:
                    start_date = datetime.strptime(date, "%Y%m").date()
                    end_date = (
                        start_date + relativedelta(months=1) - relativedelta(days=1)
                    )
                elif len(date) == 4:  # is year
                    assert splitted_filename[-2].isdigit()
                    start_date = datetime.strptime(splitted_filename[-2], "%Y").date()
                    end_date = datetime.strptime(date, "%Y").date()
                else:
                    raise ValueError(f"could not parse date {date} from file {name}")

                if not (end_date < starttime or start_date > endtime):
                    parse_files.append(file["name"].replace("_global", ""))
                    if not (
                        download_path / file["name"].replace("_global", "")
                    ).exists():
                        download_files.append(file["path"])

        if download_files:
            self.logger.info(f"Requesting download of {len(download_files)} files")
            while True:
                try:
                    response = client.cutout(download_files, [ymin, ymax, xmin, xmax])
                except requests.exceptions.HTTPError:
                    self.logger.warning(
                        "HTTPError, could not download files, retrying in 60 seconds"
                    )
                else:
                    if response["status"] == "finished":
                        break
                    elif response["status"] == "started":
                        self.logger.debug(
                            f"{response['meta']['created_files']}/{response['meta']['total_files']} files prepared on ISIMIP server for {variable}, waiting 60 seconds before retrying"
                        )
                    elif response["status"] == "queued":
                        self.logger.debug(
                            f"Data preparation queued for {variable} on ISIMIP server, waiting 60 seconds before retrying"
                        )
                    elif response["status"] == "failed":
                        self.logger.debug(
                            "ISIMIP internal server error, waiting 60 seconds before retrying"
                        )
                    else:
                        raise ValueError(
                            f"Could not download files: {response['status']}"
                        )
                time.sleep(60)
            self.logger.info(f"Starting download of files for {variable}")
            # download the file when it is ready
            client.download(
                response["file_url"], path=download_path, validate=False, extract=False
            )
            self.logger.info(f"Download finished for {variable}")
            # remove zip file
            zip_file = download_path / Path(
                urlparse(response["file_url"]).path.split("/")[-1]
            )
            # make sure the file exists
            assert zip_file.exists()
            # Open the zip file
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # Get a list of all the files in the zip file
                file_list = [f for f in zip_ref.namelist() if f.endswith(".nc")]
                # Extract each file one by one
                for i, file_name in enumerate(file_list):
                    # Rename the file
                    bounds_str = ""
                    if isinstance(ymin, float):
                        bounds_str += f"_lat{ymin}"
                    else:
                        bounds_str += f"_lat{ymin:.1f}"
                    if isinstance(ymax, float):
                        bounds_str += f"to{ymax}"
                    else:
                        bounds_str += f"to{ymax:.1f}"
                    if isinstance(xmin, float):
                        bounds_str += f"lon{xmin}"
                    else:
                        bounds_str += f"lon{xmin:.1f}"
                    if isinstance(xmax, float):
                        bounds_str += f"to{xmax}"
                    else:
                        bounds_str += f"to{xmax:.1f}"
                    assert bounds_str in file_name
                    new_file_name = file_name.replace(bounds_str, "")
                    zip_ref.getinfo(file_name).filename = new_file_name
                    # Extract the file
                    if os.name == "nt":
                        max_file_path_length = 260
                    else:
                        max_file_path_length = os.pathconf("/", "PC_PATH_MAX")
                    assert (
                        len(str(download_path / new_file_name)) <= max_file_path_length
                    ), f"File path too long: {download_path / zip_ref.getinfo(file_name).filename}"
                    zip_ref.extract(file_name, path=download_path)
            # remove zip file
            (
                download_path / Path(urlparse(response["file_url"]).path.split("/")[-1])
            ).unlink()

        datasets = [
            xr.open_dataset(
                download_path / file,
                chunks={"time": 1, "lat": XY_CHUNKSIZE, "lon": XY_CHUNKSIZE},
                lock=False,
            )
            for file in parse_files
        ]
        for dataset in datasets:
            assert "lat" in dataset.coords and "lon" in dataset.coords

        # make sure y is decreasing rather than increasing
        datasets = [
            (
                dataset.reindex(lat=dataset.lat[::-1])
                if dataset.lat[0] < dataset.lat[-1]
                else dataset
            )
            for dataset in datasets
        ]

        reference = datasets[0]
        for dataset in datasets:
            # make sure all datasets have more or less the same coordinates
            assert np.isclose(
                dataset.coords["lat"].values,
                reference["lat"].values,
                atol=abs(datasets[0].rio.resolution()[1] / 50),
                rtol=0,
            ).all()
            assert np.isclose(
                dataset.coords["lon"].values,
                reference["lon"].values,
                atol=abs(datasets[0].rio.resolution()[0] / 50),
                rtol=0,
            ).all()

        datasets = [
            ds.assign_coords(
                lon=reference["lon"].values, lat=reference["lat"].values, inplace=True
            )
            for ds in datasets
        ]
        if len(datasets) > 1:
            ds = xr.concat(datasets, dim="time")
        else:
            ds = datasets[0]

        if starttime is not None:
            ds = ds.sel(time=slice(starttime, endtime))
            # assert that time is monotonically increasing with a constant step size
            assert (
                ds.time.diff("time").astype(np.int64)
                == (ds.time[1] - ds.time[0]).astype(np.int64)
            ).all()

        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        assert not ds.lat.attrs, "lat already has attributes"
        assert not ds.lon.attrs, "lon already has attributes"
        ds.lat.attrs = {
            "long_name": "latitude of grid cell center",
            "units": "degrees_north",
        }
        ds.lon.attrs = {
            "long_name": "longitude of grid cell center",
            "units": "degrees_east",
        }
        ds = ds.rio.write_crs(4326).rio.write_coordinate_system()

        # check whether data is for noon or midnight. If noon, subtract 12 hours from time coordinate to align with other datasets
        if hasattr(ds, "time") and pd.to_datetime(ds.time[0].values).hour == 12:
            # subtract 12 hours from time coordinate
            self.logger.warning(
                "Subtracting 12 hours from time coordinate to align climate datasets"
            )
            ds = ds.assign_coords(time=ds.time - np.timedelta64(12, "h"))
        return ds

    def write_grid(self):
        self._assert_write_mode
        for var, grid in self.grid.items():
            if self.is_updated["grid"][var]["updated"]:
                self.logger.info(f"Writing {var}")
                self.model_structure["grid"][var] = var + ".tif"
                self.is_updated["grid"][var]["filename"] = var + ".tif"
                fp = Path(self.root, var + ".tif")
                fp.parent.mkdir(parents=True, exist_ok=True)
                grid.rio.to_raster(fp, compress="LZW")

    def write_subgrid(self):
        self._assert_write_mode
        for var, grid in self.subgrid.items():
            if self.is_updated["subgrid"][var]["updated"]:
                self.logger.info(f"Writing {var}")
                self.model_structure["subgrid"][var] = var + ".tif"
                self.is_updated["subgrid"][var]["filename"] = var + ".tif"
                fp = Path(self.root, var + ".tif")
                fp.parent.mkdir(parents=True, exist_ok=True)
                grid.rio.to_raster(fp)

    def write_region_subgrid(self):
        self._assert_write_mode
        for var, grid in self.region_subgrid.items():
            if self.is_updated["region_subgrid"][var]["updated"]:
                self.logger.info(f"Writing {var}")
                self.model_structure["region_subgrid"][var] = var + ".tif"
                self.is_updated["region_subgrid"][var]["filename"] = var + ".tif"
                fp = Path(self.root, var + ".tif")
                fp.parent.mkdir(parents=True, exist_ok=True)
                grid.rio.to_raster(fp)

    def write_MERIT_grid(self):
        self._assert_write_mode
        for var, grid in self.MERIT_grid.items():
            if self.is_updated["MERIT_grid"][var]["updated"]:
                self.logger.info(f"Writing {var}")
                self.model_structure["MERIT_grid"][var] = var + ".tif"
                self.is_updated["MERIT_grid"][var]["filename"] = var + ".tif"
                fp = Path(self.root, var + ".tif")
                fp.parent.mkdir(parents=True, exist_ok=True)
                grid.rio.to_raster(fp)

    def write_MODFLOW_grid(self):
        self._assert_write_mode
        for var, grid in self.MODFLOW_grid.items():
            if self.is_updated["MODFLOW_grid"][var]["updated"]:
                self.logger.info(f"Writing {var}")
                self.model_structure["MODFLOW_grid"][var] = var + ".tif"
                self.is_updated["MODFLOW_grid"][var]["filename"] = var + ".tif"
                fp = Path(self.root, var + ".tif")
                fp.parent.mkdir(parents=True, exist_ok=True)
                grid.rio.to_raster(fp)

    def write_forcing_to_netcdf(self, var, forcing) -> None:
        self.logger.info(f"Write {var}")
        fn = var + ".nc"
        self.model_structure["forcing"][var] = fn
        self.is_updated["forcing"][var]["filename"] = fn
        fp = Path(self.root, fn)
        fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists():
            fp.unlink()
        forcing = forcing.rio.write_crs(self.crs).rio.write_coordinate_system()
        forcing = forcing.chunk(
            {
                "time": 1,
                "y": min(forcing.y.size, XY_CHUNKSIZE),
                "x": min(forcing.x.size, XY_CHUNKSIZE),
            }
        )
        with ProgressBar(dt=10):  # print progress bar every 10 seconds
            assert (
                forcing.dims[0] == "time"
            ), "time dimension must be first, otherwise xarray will not chunk correctly"
            forcing.to_netcdf(
                fp,
                mode="w",
                engine="netcdf4",
                encoding={
                    forcing.name: {
                        "chunksizes": (
                            1,
                            min(forcing.y.size, XY_CHUNKSIZE),
                            min(forcing.x.size, XY_CHUNKSIZE),
                        ),
                        "zlib": True,
                        "complevel": 9,
                    }
                },
            )
            return xr.open_dataset(
                fp, chunks={"time": 1, "y": XY_CHUNKSIZE, "x": XY_CHUNKSIZE}, lock=False
            )[forcing.name]

    def write_forcing(self) -> None:
        self._assert_write_mode
        self.logger.info("Write forcing files")
        for var in self.forcing:
            forcing = self.forcing[var]
            if self.is_updated["forcing"][var]["updated"]:
                self.write_forcing_to_netcdf(var, forcing)

    def write_table(self):
        if len(self.table) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.table.items():
                if self.is_updated["table"][name]["updated"]:
                    fn = os.path.join(name + ".csv")
                    self.logger.debug(f"Writing file {fn}")
                    self.model_structure["table"][name] = fn
                    self.is_updated["table"][name]["filename"] = fn
                    self.logger.debug(f"Writing file {fn}")
                    fp = Path(self.root, fn)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    data.to_csv(fp)

    def write_binary(self):
        if len(self.binary) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.binary.items():
                if self.is_updated["binary"][name]["updated"]:
                    fn = os.path.join(name + ".npz")
                    self.logger.debug(f"Writing file {fn}")
                    self.model_structure["binary"][name] = fn
                    self.is_updated["binary"][name]["filename"] = fn
                    self.logger.debug(f"Writing file {fn}")
                    fp = Path(self.root, fn)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(fp, data=data)

    def write_dict(self):
        def convert_timestamp_to_string(timestamp):
            return timestamp.isoformat()

        if len(self.dict) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.dict.items():
                if self.is_updated["dict"][name]["updated"]:
                    fn = os.path.join(name + ".json")
                    self.model_structure["dict"][name] = fn
                    self.is_updated["dict"][name]["filename"] = fn
                    self.logger.debug(f"Writing file {fn}")
                    output_path = Path(self.root) / fn
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(data, f, default=convert_timestamp_to_string)

    def write_geoms(self, fn: str = "{name}.gpkg", **kwargs) -> None:
        """Write model geometries to a vector file (by default gpkg) at <root>/<fn>

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'geoms/{name}.gpkg'
        """
        if len(self._geoms) == 0:
            self.logger.debug("No geoms data found, skip writing.")
            return
        else:
            self._assert_write_mode
            if "driver" not in kwargs:
                kwargs.update(driver="GPKG")
            for name, gdf in self._geoms.items():
                if self.is_updated["geoms"][name]["updated"]:
                    self.logger.debug(f"Writing file {fn.format(name=name)}")
                    self.model_structure["geoms"][name] = fn.format(name=name)
                    _fn = os.path.join(self.root, fn.format(name=name))
                    if not os.path.isdir(os.path.dirname(_fn)):
                        os.makedirs(os.path.dirname(_fn))
                    self.is_updated["geoms"][name]["filename"] = _fn
                    gdf.to_file(_fn, **kwargs)

    def set_table(self, table, name, update=True):
        self.is_updated["table"][name] = {"updated": update}
        self.table[name] = table

    def set_binary(self, data, name, update=True):
        self.is_updated["binary"][name] = {"updated": update}
        self.binary[name] = data

    def set_dict(self, data, name, update=True):
        self.is_updated["dict"][name] = {"updated": update}
        self.dict[name] = data

    def write_model_structure(self):
        with open(Path(self.root, "model_structure.json"), "w") as f:
            json.dump(self.model_structure, f, indent=4, cls=PathEncoder)

    def write(self):
        self.write_geoms()
        self.write_binary()
        self.write_table()
        self.write_dict()

        self.write_grid()
        self.write_subgrid()
        self.write_region_subgrid()
        self.write_MERIT_grid()
        self.write_MODFLOW_grid()

        self.write_forcing()

        self.write_model_structure()

    def read_model_structure(self):
        model_structure_is_empty = all(
            len(v) == 0 for v in self.model_structure.values()
        )
        if model_structure_is_empty:
            with open(Path(self.root, "model_structure.json"), "r") as f:
                self.model_structure = json.load(f)

    def read_geoms(self):
        self.read_model_structure()
        for name, fn in self.model_structure["geoms"].items():
            geom = gpd.read_file(Path(self.root, fn))
            self.set_geoms(geom, name=name, update=False)

    def read_binary(self):
        self.read_model_structure()
        for name, fn in self.model_structure["binary"].items():
            binary = np.load(Path(self.root, fn))["data"]
            self.set_binary(binary, name=name, update=False)

    def read_table(self):
        self.read_model_structure()
        for name, fn in self.model_structure["table"].items():
            table = pd.read_csv(Path(self.root, fn))
            self.set_table(table, name=name, update=False)

    def read_dict(self):
        self.read_model_structure()
        for name, fn in self.model_structure["dict"].items():
            with open(Path(self.root, fn), "r") as f:
                d = json.load(f)
            self.set_dict(d, name=name, update=False)

    def read_netcdf(self, fn: str, name: str) -> xr.Dataset:
        with xr.load_dataset(Path(self.root) / fn, decode_cf=False).rename(
            {"band_data": name}
        ) as da:
            if fn.endswith(".tif") and "band" in da.dims and da.band.size == 1:
                da = da.sel(band=1)
            if fn.endswith(".tif"):
                da.x.attrs = {
                    "long_name": "latitude of grid cell center",
                    "units": "degrees_north",
                }
                da.y.attrs = {
                    "long_name": "longitude of grid cell center",
                    "units": "degrees_east",
                }
            return da

    def read_grid(self) -> None:
        for name, fn in self.model_structure["grid"].items():
            data = self.read_netcdf(fn, name=name)
            self.set_grid(data, name=name, update=False)

    def read_subgrid(self) -> None:
        for name, fn in self.model_structure["subgrid"].items():
            data = self.read_netcdf(fn, name=name)
            self.set_subgrid(data, name=name, update=False)

    def read_region_subgrid(self) -> None:
        for name, fn in self.model_structure["region_subgrid"].items():
            data = self.read_netcdf(fn, name=name)
            self.set_region_subgrid(data, name=name, update=False)

    def read_MERIT_grid(self) -> None:
        for name, fn in self.model_structure["MERIT_grid"].items():
            data = self.read_netcdf(fn, name=name)
            self.set_MERIT_grid(data, name=name, update=False)

    def read_MODFLOW_grid(self) -> None:
        for name, fn in self.model_structure["MODFLOW_grid"].items():
            data = self.read_netcdf(fn, name=name)
            self.set_MODFLOW_grid(data, name=name, update=False)

    def read_forcing(self) -> None:
        self.read_model_structure()
        for name, fn in self.model_structure["forcing"].items():
            with xr.open_dataset(
                Path(self.root) / fn,
                chunks={"time": 1, "y": XY_CHUNKSIZE, "x": XY_CHUNKSIZE},
                lock=False,
            )[name.split("/")[-1]] as da:
                assert "x" in da.dims and "y" in da.dims
                self.set_forcing(da, name=name, update=False)

    def read(self):
        self.read_model_structure()

        self.read_geoms()
        self.read_binary()
        self.read_table()
        self.read_dict()

        self.read_grid()
        self.read_subgrid()
        self.read_region_subgrid()
        self.read_MERIT_grid()
        self.read_MODFLOW_grid()

        self.read_forcing()

    def set_geoms(self, geoms, name, update=True):
        self.is_updated["geoms"][name] = {"updated": update}
        super().set_geoms(geoms, name=name)

    def set_forcing(self, data, name: str, update=True, write=True, *args, **kwargs):
        self.is_updated["forcing"][name] = {"updated": update}
        if update and write:
            data = self.write_forcing_to_netcdf(name, data)
            self.is_updated["forcing"][name]["updated"] = False
        super().set_forcing(data, name=name, *args, **kwargs)

    def _set_grid(
        self,
        grid,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        assert grid is not None
        # NOTE: variables in a dataset are not longer renamed as used to be the case in
        # set_staticmaps
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray):
            if data.shape != grid.raster.shape:
                raise ValueError("Shape of data and grid maps do not match")
            data = xr.DataArray(dims=grid.raster.dims, data=data, name=name)
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")
        # force read in r+ mode
        if len(grid) == 0:  # trigger init / read
            grid[name] = data[name]
        else:
            for dvar in data.data_vars:
                if dvar in grid:
                    if self._read:
                        self.logger.warning(f"Replacing grid map: {dvar}")
                grid[dvar] = data[dvar]
        return grid

    def set_grid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["grid"][name] = {"updated": update}
        super().set_grid(data, name=name)

    def set_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["subgrid"][name] = {"updated": update}
        self._set_grid(self.subgrid, data, name=name)

    def set_region_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["region_subgrid"][name] = {"updated": update}
        self._set_grid(self.region_subgrid, data, name=name)

    def set_MERIT_grid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["MERIT_grid"][name] = {"updated": update}
        self._set_grid(self.MERIT_grid, data, name=name)

    def set_MODFLOW_grid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["MODFLOW_grid"][name] = {"updated": update}
        self._set_grid(self.MODFLOW_grid, data, name=name)

    def set_alternate_root(self, root, mode):
        relative_path = Path(os.path.relpath(Path(self.root), root.resolve()))
        for data in self.model_structure.values():
            for name, fn in data.items():
                data[name] = relative_path / fn
        super().set_root(root, mode)

    @property
    def subgrid_factor(self):
        subgrid_factor = self.subgrid.dims["x"] // self.grid.dims["x"]
        assert subgrid_factor == self.subgrid.dims["y"] // self.grid.dims["y"]
        return subgrid_factor

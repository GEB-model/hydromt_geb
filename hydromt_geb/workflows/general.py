from typing import Any, Union
import xarray
import numpy as np
from collections.abc import Mapping

def repeat_grid(data, factor):
    return data.repeat(factor, axis=-2).repeat(factor, axis=-1)

def calculate_cell_area(affine_transform, shape):
    RADIUS_EARTH_EQUATOR = 40075017  # m
    distance_1_degree_latitude = RADIUS_EARTH_EQUATOR / 360

    height, width = shape

    lat_idx = np.arange(0, height).repeat(width).reshape((height, width))
    lat = (lat_idx + 0.5) * affine_transform.e + affine_transform.f
    width_m = distance_1_degree_latitude * np.cos(np.radians(lat)) * abs(affine_transform.a)
    height_m = distance_1_degree_latitude * abs(affine_transform.e)
    return (width_m * height_m)

def pad_xy(
    self,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    constant_values: Union[
        float, tuple[int, int], Mapping[Any, tuple[int, int]], None
    ] = None,
    return_slice: bool = False,
) -> xarray.DataArray:
    """Pad the array to x,y bounds.

    Parameters
    ----------
    minx: float
        Minimum bound for x coordinate.
    miny: float
        Minimum bound for y coordinate.
    maxx: float
        Maximum bound for x coordinate.
    maxy: float
        Maximum bound for y coordinate.
    constant_values: scalar, tuple or mapping of hashable to tuple
        The value used for padding. If None, nodata will be used if it is
        set, and np.nan otherwise.

    Returns
    -------
    :obj:`xarray.DataArray`:
        The padded object.
    """
    # pylint: disable=too-many-locals
    left, bottom, right, top = self._internal_bounds()
    resolution_x, resolution_y = self.resolution()
    y_before = y_after = 0
    x_before = x_after = 0
    y_coord: Union[xarray.DataArray, np.ndarray] = self._obj[self.y_dim]
    x_coord: Union[xarray.DataArray, np.ndarray] = self._obj[self.x_dim]

    if top - resolution_y < maxy:
        new_y_coord: np.ndarray = np.arange(bottom, maxy, -resolution_y)[::-1]
        y_before = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        top = y_coord[0]
    if bottom + resolution_y > miny:
        new_y_coord = np.arange(top, miny, resolution_y)
        y_after = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        bottom = y_coord[-1]

    if left - resolution_x > minx:
        new_x_coord: np.ndarray = np.arange(right, minx, -resolution_x)[::-1]
        x_before = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        left = x_coord[0]
    if right + resolution_x < maxx:
        new_x_coord = np.arange(left, maxx, resolution_x)
        x_after = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        right = x_coord[-1]

    if constant_values is None:
        constant_values = np.nan if self.nodata is None else self.nodata

    superset = self._obj.pad(
        pad_width={
            self.x_dim: (x_before, x_after),
            self.y_dim: (y_before, y_after),
        },
        constant_values=constant_values,  # type: ignore
    ).rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
    superset[self.x_dim] = x_coord
    superset[self.y_dim] = y_coord
    superset.rio.write_transform(inplace=True)
    if return_slice:
        return superset, {
            'x': slice(
                x_before,
                superset['x'].size - x_after
            ),
            'y': slice(
                y_before,
                superset['y'].size - y_after
            )
        }
    else:
        return superset
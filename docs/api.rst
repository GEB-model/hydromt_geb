.. currentmodule:: geb_sfincs


.. _api_reference:

=============
API reference
=============

.. _api_model:

GEB model class
==================

Initialize
----------

.. autosummary::
   :toctree: _generated/

   GEBModel

.. _components:

Setup components
----------------

.. autosummary::
   :toctree: _generated/

   GEBModel.setup_config
   GEBModel.setup_region
   GEBModel.setup_grid
   GEBModel.setup_grid_from_region
   GEBModel.setup_dep
   GEBModel.setup_mask_active
   GEBModel.setup_mask_bounds
   GEBModel.setup_manning_roughness
   GEBModel.setup_constant_infiltration
   GEBModel.setup_cn_infiltration
   GEBModel.setup_cn_infiltration_with_kr
   GEBModel.setup_subgrid
   GEBModel.setup_river_inflow
   GEBModel.setup_river_outflow
   GEBModel.setup_observation_points
   GEBModel.setup_structures
   GEBModel.setup_drainage_structures
   GEBModel.setup_waterlevel_forcing
   GEBModel.setup_waterlevel_bnd_from_mask
   GEBModel.setup_discharge_forcing
   GEBModel.setup_discharge_forcing_from_grid
   GEBModel.setup_precip_forcing
   GEBModel.setup_precip_forcing_from_grid
   GEBModel.setup_pressure_forcing_from_grid
   GEBModel.setup_wind_forcing
   GEBModel.setup_wind_forcing_from_grid
   GEBModel.setup_tiles

Attributes
----------

.. autosummary::
   :toctree: _generated/

   GEBModel.region
   GEBModel.mask
   GEBModel.crs
   GEBModel.res
   GEBModel.root
   GEBModel.config
   GEBModel.grid
   GEBModel.geoms
   GEBModel.forcing
   GEBModel.states
   GEBModel.results

High level methods
------------------

.. autosummary::
   :toctree: _generated/

   GEBModel.read
   GEBModel.write
   GEBModel.build
   GEBModel.update
   GEBModel.set_root

Low level methods
-----------------

.. autosummary::
   :toctree: _generated/

   GEBModel.update_grid_from_config
   GEBModel.update_spatial_attrs
   GEBModel.get_model_time

General methods
---------------

.. autosummary::
   :toctree: _generated/

   GEBModel.setup_config
   GEBModel.get_config
   GEBModel.set_config
   GEBModel.read_config
   GEBModel.write_config

   GEBModel.set_grid
   GEBModel.read_grid
   GEBModel.write_grid

   GEBModel.read_subgrid
   GEBModel.write_subgrid

   GEBModel.set_geoms
   GEBModel.read_geoms
   GEBModel.write_geoms

   GEBModel.set_forcing
   GEBModel.read_forcing
   GEBModel.write_forcing

   GEBModel.set_states
   GEBModel.read_states
   GEBModel.write_states

   GEBModel.set_results
   GEBModel.read_results

.. _workflows:

GEB workflows
================

.. autosummary::
   :toctree: _generated/

   workflows.merge_multi_dataarrays
   workflows.merge_dataarrays
   workflows.get_rivbank_dz
   workflows.get_river_bathymetry
   workflows.burn_river_zb
   workflows.snap_discharge
   workflows.river_boundary_points
   workflows.river_centerline_from_hydrography
   workflows.landuse
   workflows.cn_to_s
   workflows.create_topobathy_tiles
   workflows.downscale_floodmap_webmercator

.. _methods:

GEB low-level methods
========================

Input/Output methods
---------------------

.. autosummary::
   :toctree: _generated/

   utils.read_binary_map
   utils.write_binary_map
   utils.read_binary_map_index
   utils.write_binary_map_index
   utils.read_ascii_map
   utils.write_ascii_map
   utils.read_timeseries
   utils.write_timeseries
   utils.read_xy
   utils.write_xy
   utils.read_xyn
   utils.write_xyn
   utils.read_geoms
   utils.write_geoms
   utils.read_drn
   utils.write_drn
   utils.read_sfincs_map_results
   utils.read_sfincs_his_results
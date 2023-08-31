# HydroMT-GEB: GEB plugin for HydroMT

## What is the HydroMT-GEB plugin?

HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the process of building and analyzing spatial geoscientific models with a focus on water system models. It does so by automating the workflow to go from raw data to a complete model instance which is ready to run and to analyse model results once the simulation has finished. This plugin provides an implementation of the model API for the GEB model.

## What is GEB?
GEB, the Geographic Environmental and Behavioural model, is a coupled agent-based and hydrological model which can simulate millions of individual farmers and their bi-directional interaction with a fully distributed hydrological model on a powerful laptop. The model is jointly developed by IVM and IIASA. You can find full documentation [here](https://github.com/jensdebruijn/GEB).

The model is build using widely used hydrological model CWatM and the agent-based model Honeybees for high-performance ABMs in Python. Some of the behavioral components of the model are based on ADOPT.

## Why HydroMT-GEB?
Setting up hydrological models typically requires many (manual) steps to process input data and might therefore be time consuming and hard to reproduce. Especially improving models based on global geospatial datasets, which are rapidly becoming available at increasingly high resolutions, might be challenging. HydroMT-GEB aims to make the model building process fast, modular and reproducible. In the future, this tool will also  facilitate the analysis of GEB model results.

## How to use HydroMT-GEB?
A description on how to use this tool for creating GEB's input data is given in [GEB's repository](https://github.com/jensdebruijn/GEB).

## How to cite?
The following paper presents a real-world application of HydroMT-GEB:

de Bruijn, J. A., Smilovic, M., Burek, P., Guillaumot, L., Wada, Y., and Aerts, J. C. J. H.: GEB v0.1: a large-scale agent-based socio-hydrological model – simulating 10 million individual farming households in a fully distributed hydrological model, Geosci. Model Dev., 16, 2437–2454, [https://doi.org/10.5194/gmd-16-2437-2023](https://doi.org/10.5194/gmd-16-2437-2023), 2023.

## How to contribute?
If you find any issues in the code (or documentation) feel free to leave an issue on the GitHub issue tracker.
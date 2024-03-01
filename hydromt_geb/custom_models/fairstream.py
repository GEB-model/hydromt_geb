from ..geb import GEBModel as GEBModel

import random
import numpy as np
import geopandas as gpd
import gzip
import pandas as pd
from honeybees.library.raster import pixels_to_coords


class fairSTREAMModel(GEBModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_census_characteristics(
        self,
        maximum_age=85,
    ):
        n_farmers = self.binary["agents/farmers/id"].size
        farms = self.subgrid["agents/farmers/farms"]

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
            "age_class",
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
        GDL_region_per_farmer["age_household_head"] = np.full(
            len(GDL_region_per_farmer), -1, dtype=np.int32
        )
        GDL_region_per_farmer["education_level"] = np.full(
            len(GDL_region_per_farmer), -1, dtype=np.int32
        )
        for GDL_region, farmers_GDL_region in GDL_region_per_farmer.groupby("GDLcode"):
            with gzip.open(GLOPOP_S.path.format(region=GDL_region), "rb") as f:
                GLOPOP_S_region = np.frombuffer(f.read(), dtype=np.int32)

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

            # shuffle GLOPOP-S data to avoid biases in that regard
            GLOPOP_S_household_IDs = GLOPOP_S_region["household_ID"].unique()
            np.random.shuffle(GLOPOP_S_household_IDs)  # shuffle array in-place
            GLOPOP_S_region = (
                GLOPOP_S_region.set_index("household_ID")
                .loc[GLOPOP_S_household_IDs]
                .reset_index()
            )

            # Select a sample of farmers from the database. Because the households were
            # shuflled there is no need to pick random households, we can just take the first n_farmers.
            # If there are not enough farmers in the region, we need to upsample the data. In this case
            # we will just take the same farmers multiple times starting from the top.
            GLOPOP_S_household_IDs = GLOPOP_S_region["household_ID"].values

            # first we mask out all consecutive duplicates
            mask = np.concatenate(
                ([True], GLOPOP_S_household_IDs[1:] != GLOPOP_S_household_IDs[:-1])
            )
            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[mask]

            GLOPOP_S_region_sampled = []
            if GLOPOP_S_household_IDs.size < len(farmers_GDL_region):
                n_repetitions = len(farmers_GDL_region) // GLOPOP_S_household_IDs.size
                max_household_ID = GLOPOP_S_household_IDs.max()
                for i in range(n_repetitions):
                    GLOPOP_S_region_copy = GLOPOP_S_region.copy()
                    # increase the household ID to avoid duplicate household IDs. Using (i + 1) so that the original household IDs are not changed
                    # so that they can be used in the final "topping up" below.
                    GLOPOP_S_region_copy["household_ID"] = GLOPOP_S_region_copy[
                        "household_ID"
                    ] + ((i + 1) * max_household_ID)
                    GLOPOP_S_region_sampled.append(GLOPOP_S_region_copy)
                requested_farmers = (
                    len(farmers_GDL_region) % GLOPOP_S_household_IDs.size
                )
            else:
                requested_farmers = len(farmers_GDL_region)

            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[:requested_farmers]
            GLOPOP_S_region_sampled.append(
                GLOPOP_S_region[
                    GLOPOP_S_region["household_ID"].isin(GLOPOP_S_household_IDs)
                ]
            )

            GLOPOP_S_region_sampled = pd.concat(
                GLOPOP_S_region_sampled, ignore_index=True
            )
            assert GLOPOP_S_region_sampled["household_ID"].unique().size == len(
                farmers_GDL_region
            )

            households_region = GLOPOP_S_region_sampled.groupby("household_ID")
            # select only household heads
            household_heads = households_region.apply(
                lambda x: x[x["relation_to_household_head"] == 1]
            )
            assert len(household_heads) == len(farmers_GDL_region)

            # age
            household_heads["age"] = np.full(len(household_heads), -1, dtype=np.int32)
            age_class_to_age = {
                1: (0, 16),
                2: (16, 26),
                3: (26, 36),
                4: (36, 46),
                5: (46, 56),
                6: (56, 66),
                7: (66, maximum_age + 1),
            }  # exclusive
            for age_class, age_range in age_class_to_age.items():
                household_heads_age_class = household_heads[
                    household_heads["age_class"] == age_class
                ]
                household_heads.loc[household_heads_age_class.index, "age"] = (
                    np.random.randint(
                        age_range[0],
                        age_range[1],
                        size=len(household_heads_age_class),
                        dtype=GDL_region_per_farmer["age_household_head"].dtype,
                    )
                )
            GDL_region_per_farmer.loc[
                farmers_GDL_region.index, "age_household_head"
            ] = household_heads["age"].values

            # education level
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "education_level"] = (
                household_heads["education_level"].values
            )

            # household size
            household_sizes_region = households_region.size().values.astype(np.int32)
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "household_size"] = (
                household_sizes_region
            )

        # assert none of the household sizes are placeholder value -1
        assert (GDL_region_per_farmer["household_size"] != -1).all()
        assert (GDL_region_per_farmer["age_household_head"] != -1).all()
        assert (GDL_region_per_farmer["education_level"] != -1).all()

        self.set_binary(
            GDL_region_per_farmer["household_size"].values,
            name="agents/farmers/household_size",
        )
        self.set_binary(
            GDL_region_per_farmer["age_household_head"].values,
            name="agents/farmers/age_household_head",
        )
        self.set_binary(
            GDL_region_per_farmer["education_level"].values,
            name="agents/farmers/education_level",
        )

    def setup_farmer_characteristics(
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
        farms = self.subgrid["agents/farmers/farms"]

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

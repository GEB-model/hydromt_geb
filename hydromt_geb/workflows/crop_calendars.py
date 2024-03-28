import numpy as np


def parse_MIRCA2000_crop_calendar(data_catalog, bounds):
    MIRCA2000_unit_grid = data_catalog.get_rasterdataset(
        "MIRCA2000_unit_grid", bbox=bounds
    )
    rainfed_crop_calendar_fp = data_catalog.get_source(
        "MIRCA2000_cropping_calendar_rainfed"
    ).path

    unique_MIRCA2000_units = np.unique(MIRCA2000_unit_grid.values)

    MIRCA2000_data = {}
    with open(rainfed_crop_calendar_fp, "r") as f:
        lines = f.readlines()
        # remove all empty lines
        lines = [line.strip() for line in lines if line.strip()]
        # skip header
        lines = lines[4:]
        for line in lines:
            line = line.replace("  ", " ").split(" ")
            unit_code = int(line[0])
            if unit_code not in unique_MIRCA2000_units:
                continue
            if unit_code not in MIRCA2000_data:
                MIRCA2000_data[unit_code] = []
            crop_class = int(line[1])
            number_of_rotations = int(line[2])
            if number_of_rotations == 0:
                continue
            crops = line[3:]
            crop_rotations = []
            for rotation in range(number_of_rotations):
                area = float(crops[rotation * 3])
                if area == 0:
                    continue
                start_month = int(crops[rotation * 3 + 1])
                end_month = int(crops[rotation * 3 + 2])
                crop_rotations.append((start_month, end_month, area))

            crop_rotations = sorted(crop_rotations, key=lambda x: x[2])  # sort by area
            if len(crop_rotations) == 1:
                start_month, end_month, area = crop_rotations[0]
                crop_rotation = area, crop_class, ((start_month, end_month),)
                MIRCA2000_data[unit_code].append(crop_rotation)
            elif len(crop_rotations) == 2:
                crop_rotation = (
                    crop_rotations[1][2] - crop_rotations[0][2],
                    crop_class,
                    ((crop_rotations[1][0], crop_rotations[1][1]),),
                )
                MIRCA2000_data[unit_code].append(crop_rotation)
                crop_rotation = (
                    crop_rotations[0][2],
                    crop_class,
                    (
                        (crop_rotations[0][0], crop_rotations[0][1]),
                        (crop_rotations[1][0], crop_rotations[1][1]),
                    ),
                )
                MIRCA2000_data[unit_code].append(crop_rotation)
            else:
                raise NotImplementedError

    return MIRCA2000_data
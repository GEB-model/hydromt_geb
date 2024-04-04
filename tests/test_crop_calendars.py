from datetime import date

from hydromt_geb.workflows.crop_calendars import (
    get_growing_season_length,
    get_day_of_year,
)


def test_growing_season_length():
    assert get_growing_season_length(1, 365) == 364
    assert get_growing_season_length(1, 1) == 365
    assert get_growing_season_length(1, 2) == 1
    assert get_growing_season_length(300, 1) == 66
    assert get_growing_season_length(2, 1) == 364


def test_day_of_year():
    assert get_day_of_year(date(2000, 1, 1)) == 1
    assert get_day_of_year(date(2000, 1, 2)) == 2
    assert get_day_of_year(date(2000, 2, 1)) == 32

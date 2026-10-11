"""Chart picked for each result shape (src/ui/charts.py)."""

import pytest

from ui.charts import pick_visualization


def chart(columns, rows):
    return pick_visualization(columns, rows)["chart"]


def view(chart, key="measures"):
    """(labels, {series name: values}) of the first measure of a chart, or of its timeline."""
    data = chart[key][0]
    return data["labels"], {s["name"]: s["values"] for s in data["series"]}


# ── Comparisons: one series per year, whatever the SQL shape ─────────────────

MONTHLY_2022_2023 = {"2022": [10.0, 20.0, 30.0], "2023": [11.0, 21.0, 31.0]}


@pytest.mark.parametrize("columns, rows", [
    # GROUP BY year, month
    (["annee", "mois", "ca"], [[y, m, v] for y, vs in [(2022, [10.0, 20.0, 30.0]), (2023, [11.0, 21.0, 31.0])] for m, v in zip([1, 2, 3], vs)]),
    # Months as DATE_FORMAT(d, 'MM') strings, rows in any order
    (["year", "month", "ca"], [[2023, "03", 31.0], [2022, "01", 10.0], [2023, "01", 11.0], [2022, "03", 30.0], [2022, "02", 20.0], [2023, "02", 21.0]]),
    # One column per year (CASE WHEN pivot); the variation % stays in the table only
    (["mois", "ca_2022", "ca_2023", "evolution"], [["01", 10.0, 11.0, 0.1], ["02", 20.0, 21.0, 0.05], ["03", 30.0, 31.0, 0.03]]),
    # Year first in the column names
    (["mois", "2022_ca", "2023_ca"], [[1, 10.0, 11.0], [2, 20.0, 21.0], [3, 30.0, 31.0]]),
    # Month dates next to their year: YEAR(d), DATE_FORMAT(d, 'yyyy-MM')
    (["annee", "mois", "ca"], [[y, f"{y}-0{m}", v] for y, vs in [(2022, [10.0, 20.0, 30.0]), (2023, [11.0, 21.0, 31.0])] for m, v in zip([1, 2, 3], vs)]),
])
def test_years_compared_by_month(columns, rows):
    c = chart(columns, rows)
    assert c["type"] == "line"
    assert c["measures"][0]["name"] == "ca"
    assert view(c) == (["janv.", "févr.", "mars"], MONTHLY_2022_2023)


def test_year_comparison_offers_a_timeline():
    c = chart(["annee", "mois", "ca"], [[y, m, float(y - 2000 + m)] for y in (2022, 2023) for m in (1, 2)])
    assert view(c, "timeline") == (
        ["janv. 2022", "févr. 2022", "janv. 2023", "févr. 2023"],
        {None: [23.0, 24.0, 24.0, 25.0]},
    )


def test_missing_months_leave_gaps():
    # The current year, still running
    rows = [[2022, m, 1.0] for m in (1, 2, 3)] + [[2023, m, 2.0] for m in (1, 2)]
    assert view(chart(["annee", "mois", "ca"], rows))[1] == {"2022": [1.0, 1.0, 1.0], "2023": [2.0, 2.0, None]}


def test_quarters_by_year():
    c = chart(["trimestre", "annee", "ca"], [[q, y, 1.0] for y in (2023, 2022) for q in (2, 1)])
    labels, series = view(c)
    assert labels == ["T1", "T2"]
    assert list(series) == ["2022", "2023"]


def test_countries_by_month():
    rows = [[f"2023-0{m}-01T00:00:00", country, 1.0] for m in (1, 2) for country in ("France", "Maroc")]
    c = chart(["month_start", "country", "total"], rows)
    assert c["type"] == "line"
    assert view(c)[0] == ["2023-01-01", "2023-02-01"]
    assert list(view(c)[1]) == ["France", "Maroc"]
    assert c["timeline"] is None


def test_categories_by_year_are_grouped_bars_largest_first():
    rows = [[genre, y, v] for genre, vs in [("Jazz", (1.0, 2.0)), ("Rock", (5.0, 6.0)), ("Pop", (3.0, 1.0))] for y, v in zip((2022, 2023), vs)]
    c = chart(["genre", "annee", "ventes"], rows)
    assert c["type"] == "bar"
    assert view(c) == (["Rock", "Pop", "Jazz"], {"2022": [5.0, 3.0, 1.0], "2023": [6.0, 1.0, 2.0]})


# ── Not comparisons: a single series ─────────────────────────────────────────


@pytest.mark.parametrize("columns, rows", [
    # A hierarchy: each city belongs to one country
    (["country", "city", "total"], [["FR", "Paris", 1.5], ["FR", "Lyon", 2.5], ["MA", "Rabat", 3.5], ["MA", "Fes", 1.0]]),
    # A person's name
    (["FirstName", "LastName", "total"], [["Ana", "Silva", 1.5], ["Luis", "Silva", 2.5], ["Marc", "Costa", 3.5], ["Eva", "Costa", 1.0]]),
    # Too many years for one color each
    (["year", "month", "ca"], [[y, m, 1.5] for y in range(2010, 2020) for m in (1, 2)]),
    # A single year
    (["year", "month", "ca"], [[2023, m, 1.5] for m in (1, 2, 3)]),
])
def test_single_series(columns, rows):
    assert list(view(chart(columns, rows))[1]) == [None]


def test_month_dates_alone_are_a_timeline():
    c = chart(["mois", "ca"], [[f"{y}-0{m}", 1.0] for y in (2022, 2023) for m in (1, 2)])
    assert view(c) == (["2022-01", "2022-02", "2023-01", "2023-02"], {None: [1.0, 1.0, 1.0, 1.0]})


def test_single_year_is_a_timeline_in_time_order():
    c = chart(["year", "month", "ca"], [[2023, 2, 2.0], [2023, 1, 1.0], [2023, 3, 3.0]])
    assert c["type"] == "line"
    assert view(c) == (["janv. 2023", "févr. 2023", "mars 2023"], {None: [1.0, 2.0, 3.0]})


def test_categories_are_ranked_bars():
    c = chart(["genre", "total"], [["Jazz", 1.5], ["Rock", 9.5], ["Pop", 4.0]])
    assert c["type"] == "bar"
    assert view(c) == (["Rock", "Pop", "Jazz"], {None: [9.5, 4.0, 1.5]})


def test_ids_label_the_bars_when_nothing_else_does():
    assert view(chart(["CustomerId", "total"], [[1, 1.5], [2, 2.5]]))[0] == ["2", "1"]


def test_unrelated_measures_are_switched_not_overlaid():
    c = chart(["mois", "ca", "nb_commandes"], [[m, 1.5, 3] for m in (1, 2)])
    assert [m["name"] for m in c["measures"]] == ["ca", "nb_commandes"]
    assert all(len(m["series"]) == 1 for m in c["measures"])


# ── KPIs and no chart ────────────────────────────────────────────────────────


def test_single_row_is_kpi_tiles():
    viz = pick_visualization(["mois", "ca_2022", "ca_2023"], [[1, 10.0, 11.0]])
    assert viz["chart"] is None
    assert viz["kpis"] == [
        {"label": "ca_2022", "value": 10.0, "caption": "janv."},
        {"label": "ca_2023", "value": 11.0, "caption": "janv."},
    ]


@pytest.mark.parametrize("columns, rows", [
    ([], []),
    (["genre"], [["Rock"], ["Jazz"]]),
])
def test_nothing_to_draw(columns, rows):
    assert pick_visualization(columns, rows) == {"chart": None, "kpis": None}

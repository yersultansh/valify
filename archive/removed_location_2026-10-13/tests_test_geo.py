"""Spatial features: distances, POI parsing/download (mocked), POI cache, CLI. No network."""
import numpy as np
import pandas as pd
import pytest
import requests

from backend import config, geo, storage

CENTER = (43.25, 76.95)


def node(lat, lon, **tags):
    return {"type": "node", "id": 1, "lat": lat, "lon": lon, "tags": tags}


def way(lat, lon, **tags):
    return {"type": "way", "id": 2, "center": {"lat": lat, "lon": lon}, "tags": tags}


# ------------------------------- distances -------------------------------
def test_haversine_known_distances():
    one_deg = geo.haversine_matrix(np.array([43.0]), np.array([76.9]), np.array([[44.0, 76.9]]))[0, 0]
    assert one_deg == pytest.approx(111.19, abs=0.1)
    assert geo._hav(43.0, 76.9, 44.0, 76.9) == pytest.approx(one_deg)
    assert geo.haversine_matrix(np.array([43.25]), np.array([76.95]), np.array([[43.25, 76.95]]))[0, 0] == pytest.approx(0)


def test_count_within_radius_boundary_and_empty():
    km = 1 / 111.195                                         # 1 km in degrees of latitude
    pts = [[CENTER[0] + 1.5 * km, CENTER[1]], [CENTER[0] + 2.5 * km, CENTER[1]], [CENTER[0] - 0.5 * km, CENTER[1]]]
    assert geo.count_within([CENTER[0]], [CENTER[1]], pts)[0] == 2            # 1.5 km and 0.5 km, not 2.5 km
    assert geo.count_within([CENTER[0]], [CENTER[1]], pts, radius_km=1.0)[0] == 1
    assert geo.count_within([CENTER[0]], [CENTER[1]], [])[0] == 0
    assert np.isnan(geo.nearest_km([CENTER[0]], [CENTER[1]], [])[0])


def test_nearest_km_and_chunking():
    pts = [[43.30, 76.95], [43.26, 76.95]]
    assert geo.nearest_km([43.25], [76.95], pts)[0] == pytest.approx(1.112, abs=0.01)
    lat = np.full(1200, 43.25)                                  # more rows than one 500-row chunk
    assert geo.count_within(lat, np.full(1200, 76.95), pts).tolist() == [1] * 1200
    assert len(geo.nearest_km(lat, np.full(1200, 76.95), pts)) == 1200


def test_add_geo_features_handles_missing_coordinates():
    poi = {"schools": [[43.255, 76.95]] * 3, "hospitals": [], "kindergartens": [[43.26, 76.95]],
           "metro": [[43.30, 76.95]]}
    df = pd.DataFrame({"latitude": [43.25, None, "bad", 43.25], "longitude": [76.95, 76.95, 76.95, None]})
    out = geo.add_geo_features(df, poi)
    assert out.loc[0, "schools_within_2km"] == 3 and out.loc[0, "hospitals_within_2km"] == 0
    assert out.loc[0, "kindergartens_within_2km"] == 1 and out.loc[0, "dist_to_metro"] == pytest.approx(5.56, abs=0.05)
    assert out.loc[1:, geo.GEO_FEATURES].isna().all().all()                    # no coordinates -> NaN
    no_cols = geo.add_geo_features(pd.DataFrame({"x": [1]}), poi)
    assert no_cols[geo.GEO_FEATURES].isna().all().all() and "latitude" not in no_cols.columns


# ------------------------------- Overpass parsing -------------------------------
def sample_response():
    return {"elements": [
        node(43.2500, 76.9500, amenity="school", name="Школа №1"),
        way(43.25005, 76.95005, amenity="school", name="Школа №1"),               # same school mapped twice
        node(43.2600, 76.9500, amenity="school"),
        node(43.26001, 76.95001, amenity="school"),                                # unnamed duplicate (< 20 m)
        node(43.2700, 76.9500, amenity="kindergarten", **{"name:ru": "Детский сад"}),
        way(43.2800, 76.9500, amenity="hospital", name="Больница"),
        node(43.2900, 76.9500, healthcare="hospital", name="Клиника"),
        node(43.1900, 76.9000, railway="station", station="subway", name="Сайран"),
        node(43.19001, 76.90001, public_transport="station", subway="yes", name="Сайран"),   # duplicate
        node(43.2300, 76.9400, railway="station", station="subway", name="Абай"),
        node(43.3000, 76.9500, amenity="cafe", name="not relevant"),
        {"type": "relation", "id": 3, "tags": {"amenity": "school"}},               # no coordinates: skipped
    ]}


def test_parse_overpass_categories_and_dedup():
    poi = geo.parse_overpass(sample_response(), (43.1, 76.8, 43.4, 77.1))
    assert len(poi["schools"]) == 2 and len(poi["kindergartens"]) == 1 and len(poi["hospitals"]) == 2
    assert len(poi["metro"]) == 2 and sorted(poi["metro_names"]) == ["абай", "сайран"]
    assert poi["bbox"] == [43.1, 76.8, 43.4, 77.1] and "OpenStreetMap" in poi["source"] and poi["fetched_at"]


def test_parse_overpass_requires_metro():
    with pytest.raises(geo.GeoError, match="metro"):
        geo.parse_overpass({"elements": [node(43.25, 76.95, amenity="school")]}, (0, 0, 1, 1))


def test_overpass_query_uses_city_wide_box_for_metro():
    q = geo.overpass_query((43.1, 76.8, 43.4, 77.1))
    assert '"amenity"="school"](43.1,76.8,43.4,77.1)' in q
    assert f'["station"="subway"]({",".join(str(v) for v in geo.ALMATY_BBOX)})' in q and q.endswith("out center tags;")


def test_bbox_for_adds_margin_and_has_default():
    df = pd.DataFrame({"latitude": [43.2, 43.3], "longitude": [76.8, 77.0]})
    assert geo.bbox_for(df) == (43.16, 76.76, 43.34, 77.04)
    assert geo.bbox_for(pd.DataFrame({"x": [1]})) == (43.06, 76.66, 43.49, 77.19)


# ------------------------------- download -------------------------------
class Resp:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code, self._payload, self._bad = status, payload, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._payload


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append((url, data, headers, timeout))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_fetch_poi_falls_back_to_second_mirror():
    s = Session(Resp(504), Resp(200, sample_response()))
    poi = geo.fetch_poi((43.1, 76.8, 43.4, 77.1), session=s)
    assert len(poi["metro"]) == 2 and [c[0] for c in s.calls] == list(geo.OVERPASS_URLS)
    assert s.calls[0][2]["User-Agent"] == geo.USER_AGENT and "data" in s.calls[0][1]


def test_fetch_poi_handles_network_errors_and_bad_json():
    s = Session(requests.ConnectionError("down"), Resp(200, bad_json=True))
    with pytest.raises(geo.GeoError, match="Overpass request failed"):
        geo.fetch_poi((0, 0, 1, 1), session=s)
    assert len(s.calls) == 2                                                    # exactly one try per mirror


# ------------------------------- cache + CLI -------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    csv = tmp_path / "data.csv"
    pd.DataFrame({"listing_id": ["1", "2"], "latitude": [43.2, 43.3], "longitude": [76.8, 76.9]}).to_csv(csv, index=False)
    return csv


def test_poi_cache_roundtrip(env):
    assert storage.load_poi() is None
    storage.save_poi({"metro": [[43.2, 76.9]], "schools": []})
    assert storage.load_poi()["metro"] == [[43.2, 76.9]]
    storage._poi_path().write_text("{broken", encoding="utf-8")
    assert storage.load_poi() is None
    storage.save_poi({"schools": [[1, 2]]})                                     # no metro -> unusable
    assert storage.load_poi() is None


def test_cli_fetch_and_check(env, monkeypatch, capsys):
    assert geo.main(["--check", "--input", str(env)]) == 1                      # no POI yet
    monkeypatch.setattr(geo, "fetch_poi", lambda bbox, **kw: geo.parse_overpass(sample_response(), bbox))
    assert geo.main(["--fetch", "--input", str(env)]) == 0
    out = capsys.readouterr().out
    assert "2 with coordinates" in out and "Метро" not in out and "OpenStreetMap" in out and storage.load_poi()
    assert geo.main(["--check", "--input", str(env)]) == 0
    assert "dist_to_metro" in capsys.readouterr().out


def test_cli_fetch_failure_exit_code(env, monkeypatch, capsys):
    def boom(bbox, **kw):
        raise geo.GeoError("down")
    monkeypatch.setattr(geo, "fetch_poi", boom)
    assert geo.main(["--fetch", "--input", str(env)]) == 1 and "FAILED" in capsys.readouterr().out
    assert storage.load_poi() is None

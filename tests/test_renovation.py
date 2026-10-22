"""Finish flag regression tests. Synthetic prices are test-only, not market premiums."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend import batch, config, main, metrics, parser, service, storage
from tests.test_service import FLAT, frame, loader_for

RETIRED = {'dist_to_metro','schools_within_2km','hospitals_within_2km',
           'kindergartens_within_2km','latitude','longitude','geo_imputed'}

@pytest.mark.parametrize('value,expected', [(0,0),(1,1),('0',0),('1',1),
    ('Черновая отделка',0),('черновая отделка',0),('Другое',1),('свежий ремонт',1),('не указано',1)])
def test_flag_mapping(value, expected):
    assert metrics._renovation_flag(pd.Series([value])).iloc[0] == expected

def test_missing_flag_and_repeat_features():
    v=metrics._renovation_flag(pd.Series([None,np.nan,0,1]))
    assert v.iloc[:2].isna().all()
    assert metrics._renovation_flag(v).iloc[2:].tolist() == [0,1]
    f=metrics.add_features(pd.DataFrame([dict(FLAT,renovation=0)]))
    assert metrics.add_features(f).renovation.iloc[0] == 0

def renovation_frame():
    base=frame(90,seed=22)
    shell=base.assign(renovation='черновая отделка',price_kzt=base.price_kzt*0.65)
    finished=base.assign(renovation='свежий ремонт')
    finished['listing_id']=finished.listing_id.astype(str)+'9'
    return pd.concat([shell,finished],ignore_index=True)

@pytest.fixture(scope='module')
def renovation_service(tmp_path_factory):
    mp=pytest.MonkeyPatch()
    mp.setattr(storage,'MODEL_PATH',tmp_path_factory.mktemp('finish')/'model.joblib')
    s=service.PriceService(loader=loader_for(renovation_frame()),use_saved_model=False)
    s.ensure_ready()
    yield s
    mp.undo()

def test_model_learns_finish_effect(renovation_service):
    s=renovation_service
    assert 'renovation' in s.metrics['features']
    assert not RETIRED.intersection(s.metrics['features'])
    assert s.predict(dict(FLAT,renovation=1))['price_estimate_kzt'] > s.predict(dict(FLAT,renovation=0))['price_estimate_kzt']*1.10

def test_api_and_interface_keep_finish_remove_location(renovation_service):
    c=TestClient(main.create_app(svc=renovation_service,warmup=False))
    shell=c.post('/api/predict',json=dict(FLAT,renovation=0))
    finished=c.post('/api/predict',json=dict(FLAT,renovation=1))
    assert shell.status_code == finished.status_code == 200
    assert shell.json()['price_estimate_kzt'] < finished.json()['price_estimate_kzt']
    for label,expected in [('Черновая отделка',shell),('Другое',finished)]:
        assert c.post('/api/predict',json=dict(FLAT,renovation=label)).json()['price_estimate_kzt'] == expected.json()['price_estimate_kzt']
    for flag in [-1,2,0.5]:
        r=c.post('/api/predict',json=dict(FLAT,renovation=flag))
        assert r.status_code == 422 and 'отделки' in r.json()['message']
    legacy=c.post('/api/predict',json=dict(FLAT,renovation=1,**dict.fromkeys(RETIRED,999)))
    assert legacy.json() == finished.json()
    assert 'spatial_features_used' not in finished.json()
    schema=c.get('/openapi.json').json()['components']['schemas']['PredictRequest']['properties']
    assert 'renovation' in schema and not RETIRED.intersection(schema)
    assert 'limits' not in c.get('/api/options').json()
    html=c.get('/').text; js=c.get('/static/app.js').text
    assert 'name="renovation" value="0"' in html and 'name="renovation" value="1"' in html
    assert 'Черновая отделка' in html and 'Другое' in html and 'renovation: Number(' in js
    assert all(k not in html+js for k in RETIRED)
    assert not any('Пространствен' in w or 'типичные значения' in w for w in finished.json()['warnings'])

def test_legacy_fields_do_not_affect_training_or_fingerprint():
    raw=renovation_frame(); before=raw.copy(deep=True)
    a=metrics.add_features(raw); b=metrics.add_features(raw.assign(**dict.fromkeys(RETIRED,123.0)))
    pd.testing.assert_frame_equal(a[metrics.FEATURE_COLUMNS],b[metrics.FEATURE_COLUMNS])
    assert service.PriceService._fingerprint(a) == service.PriceService._fingerprint(b)
    pd.testing.assert_frame_equal(raw,before)
    for field in ['renovation','price_kzt','area_total','house_year']:
        changed=raw.copy()
        changed.loc[0,field]='свежий ремонт' if field=='renovation' else changed.loc[0,field]+1
        assert service.PriceService._fingerprint(metrics.add_features(changed)) != service.PriceService._fingerprint(a)

def test_old_model_cache_is_rebuilt(monkeypatch,tmp_path):
    monkeypatch.setattr(storage,'MODEL_PATH',tmp_path/'model.joblib')
    storage.save_model({'fingerprint':'old-location-version','model':None,'metrics':{},'trained_at':'old'})
    s=service.PriceService(loader=loader_for(frame(60)))
    s.ensure_ready()
    assert s.model is not None and s.trained_at!='old'
    assert not RETIRED.intersection(s.metrics['features'])

def test_numeric_missing_values_still_work(renovation_service):
    assert metrics.predict_with_range(renovation_service.model,dict(FLAT,house_year=None,building_type=None,renovation=None))[0]>0

def test_retired_commands_and_coordinate_parsing_are_gone(monkeypatch):
    monkeypatch.setattr(batch,'make_store',lambda *_: pytest.fail('Retired CLI must not open storage'))
    with pytest.raises(SystemExit) as exc:
        batch.main(['--enrich'])
    assert exc.value.code==2
    result=parser.parse_listing_detail((Path(__file__).parent/'fixtures/detail.html').read_text())
    assert 'json_lat' not in result and 'json_lon' not in result
    assert not hasattr(storage,'load_poi') and not hasattr(storage,'save_poi')

def test_relative_store_stays_in_project(monkeypatch,tmp_path):
    monkeypatch.setattr(config,'BASE_DIR',tmp_path)
    monkeypatch.setenv('LOCAL_STORE_DIR','data/collected')
    batch.make_store(batch.Settings(backend='local')).prepare()
    assert (tmp_path/'data/collected').is_dir()

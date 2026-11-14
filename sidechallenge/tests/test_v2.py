"""Regression tests for analytical memory, evidence and the opportunity radar."""
from fastapi.testclient import TestClient
from sidechallenge import service


def test_followup_weekend_keeps_metrics_without_second_llm(monkeypatch):
    async def fail_if_called(*args):
        raise AssertionError("Contextual refinement must not call Helmcode")
    monkeypatch.setattr(service,"_llm_plan",fail_if_called)
    with TestClient(service.app) as client:
        x=client.post('/api/assistant',json={
          'message':'Ara només els caps de setmana',
          'previous_query':'Heatmap de preu i CO2',
          'previous_plan':{'kind':'heatmap','metrics':['precio_eur_per_kwh','co2_kg_per_kwh'],
                           'group':'weekday','aggregation':'mean'}})
        assert x.status_code==200
        result=x.json()
        assert result['mode']=='context'
        assert result['day_type']=='weekend'
        assert len(result['charts'])==2
        assert 0 < result['evidence']['rows'] < 5993
        assert 'Filtre aplicat' in result['insights'][0]


def test_chart_type_refinement(monkeypatch):
    async def forbidden(*args): raise AssertionError('unnecessary call')
    monkeypatch.setattr(service,'_llm_plan',forbidden)
    with TestClient(service.app) as client:
        result=client.post('/api/assistant',json={
          'message':'Ara canvia-ho a barres i fes la mediana per mes',
          'previous_query':'Preus i emissions',
          'previous_plan':{'kind':'heatmap','metrics':['precio_eur_per_kwh'],
                           'group':'weekday','aggregation':'mean'}}).json()
        assert result['mode']=='context'
        assert result['plan']['kind']=='bar'
        assert result['plan']['group']=='month'
        assert result['plan']['aggregation']=='median'
        assert len(result['charts'])==1


def test_explicit_new_metric_overrides_previous(monkeypatch):
    async def model_plan(query,fields,has_time): return None,'Prova sense Helmcode'
    monkeypatch.setattr(service,'_llm_plan',model_plan)
    with TestClient(service.app) as client:
        result=client.post('/api/assistant',json={
          'message':'Fes-me un histograma de la demanda',
          'previous_query':'Heatmap de preus',
          'previous_plan':{'kind':'heatmap','metrics':['precio_eur_per_kwh'],
                           'group':'weekday','aggregation':'mean'}}).json()
        assert result['plan']['metrics']==['load_kw']
        assert result['plan']['kind']=='histogram'


def test_comparison_is_fair_and_uses_same_axis(monkeypatch):
    async def forbidden(*args): raise AssertionError('unnecessary model call')
    monkeypatch.setattr(service,'_llm_plan',forbidden)
    with TestClient(service.app) as client:
        r=client.post('/api/assistant',json={
          'message':'Ara compara-ho amb els laborables',
          'previous_query':'Heatmap de preus i emissions',
          'previous_plan':{'kind':'heatmap','metrics':['precio_eur_per_kwh','co2_kg_per_kwh'],
                           'group':'weekday','aggregation':'mean'}})
        assert r.status_code==200
        d=r.json()
        assert d['comparison']
        assert d['mode']=='context'
        assert len(d['charts'])==2
        assert all(len(fig['figure']['data'])==2 for fig in d['charts'])
        assert all(len(fig['figure']['data'][0]['x'])==24 for fig in d['charts'])
        assert d['evidence']['groups']['weekend']['rows'] > 0
        assert d['evidence']['groups']['weekday']['rows'] > 0


def test_untrusted_previous_fields_cannot_be_executed(monkeypatch):
    async def local_plan(*args):return None,'No LLM available'
    monkeypatch.setattr(service,'_llm_plan',local_plan)
    with TestClient(service.app) as client:
        d=client.post('/api/assistant',json={
            'message':'ara igual que abans',
            'previous_plan':{'kind':'heatmap','metrics':['__class__','__proto__'],'group':'weekday','aggregation':'mean'}}).json()
        assert d['mode']=='local'
        assert '__class__' not in d['plan']['metrics']
        assert '__proto__' not in d['plan']['metrics']


def test_historical_radar_has_physical_margin_and_grounded_disclaimer():
    with TestClient(service.app) as client:
        r=client.get('/api/radar')
        assert r.status_code==200
        d=r.json()
        assert len(d['profile'])==24
        candidate=d['candidate']
        assert 0 <= candidate['charge_hour'] < candidate['discharge_hour'] <= 23
        assert candidate['days_examined']>10
        assert 0<=candidate['viable_day_pct']<=100
        assert 'no' in d['disclaimer'].lower()


def test_v2_html_has_board_and_lab_entrypoints():
    from pathlib import Path
    html=(Path(service.HERE)/'static/index.html').read_text()
    js=(Path(service.HERE)/'static/app.js').read_text()
    assert 'id="view-board"' in html
    assert 'id="board-export"' in html
    assert 'id="lab-pin"' in html
    assert 'id="radar-apply"' in html
    assert 'previous_plan:context?.plan' in js
    assert 'sessionStorage.setItem' in js

"""Run: python -m pytest sidechallenge/tests -q"""
from io import BytesIO

from fastapi.testclient import TestClient

from sidechallenge.service import app


def test_status_and_overview():
    with TestClient(app) as c:
        r = c.get('/api/status')
        assert r.status_code == 200
        assert any(d['id'] == 'campus' for d in r.json()['datasets'])
        o = c.get('/api/overview').json()
        assert o['rows'] > 100
        assert o['kpi']['demand_mwh'] > 0
        assert len(o['profile']) == 24


def test_chart_and_local_assistant():
    with TestClient(app) as c:
        r = c.post('/api/chart', json={'kind':'heatmap', 'metrics':['precio_eur_per_kwh','co2_kg_per_kwh']})
        assert r.status_code == 200
        assert len(r.json()['charts']) == 2
        r = c.post('/api/assistant', json={'message':'Fes un heatmap del preu i CO2 per hora i dia de la setmana'})
        assert r.status_code == 200
        assert r.json()['plan']['kind'] == 'heatmap'
        assert len(r.json()['charts']) == 2
        assert r.json()['insights']


def test_retro_dispatch_is_cyclic():
    with TestClient(app) as c:
        date=c.get('/api/days').json()['dates'][0]
        r=c.post('/api/simulate',json={'date':date, 'charge_hour':14, 'discharge_hour':20})
        assert r.status_code == 200
        x=r.json()
        assert abs(x['manual']['soc_end_kwh'] - 1000.0) < .01
        assert abs(x['optimal']['soc_end_kwh'] - 1000.0) < .01
        assert len(x['manual_actions']) == 24
        assert min(x['manual_soc']) >= 0
        assert max(x['manual_soc']) <= 2000
        assert min(x['manual_actions']) >= -500
        assert max(x['manual_actions']) <= 500


def test_uploaded_files_isolated_by_session():
    content=b'Fecha,x,y\n2025-01-01,1,2\n2025-01-02,2,5\n2025-01-03,3,2\n2025-01-04,4,4\n2025-01-05,5,5\n'
    with TestClient(app) as c:
        r=c.post('/api/upload',files={'file':('test.csv',BytesIO(content),'text/csv')})
        assert r.status_code == 200
        dsid=r.json()['id']
        assert c.get('/api/overview',params={'dataset_id':dsid}).status_code == 200
    with TestClient(app) as stranger:
        assert stranger.get('/api/overview',params={'dataset_id':dsid}).status_code == 404


def test_assistant_what_if_uses_simulator():
    with TestClient(app) as c:
        r=c.post('/api/assistant',json={'message':'Què passaria si carrego a les 14 i descarrego a les 20?'})
        assert r.status_code == 200
        x=r.json()
        assert x['mode'] == 'physics'
        sim = x['evidence']['items'][0]['summary']
        assert sim['charge_window'] == [14, 14] and sim['discharge_window'] == [20, 20]
        assert abs(sim['soc_end_kwh'] - 1000) < 0.01   # closed daily cycle
        assert len(x['charts']) == 1
        assert x['verification']['ok']


def test_helmcode_fallback_is_explained(monkeypatch):
    """The UI must not claim an LLM was used after a failed provider call."""
    async def failing_plan(query, fields, has_time):
        return None, "Model o endpoint no trobat (HTTP 404)."
    monkeypatch.setattr('sidechallenge.service._llm_plan', failing_plan)
    with TestClient(app) as c:
        r = c.post('/api/assistant', json={'message':'Heatmap dels preus i emissions'})
        assert r.status_code == 200
        assert r.json()['mode'] == 'local'
        assert '404' in r.json()['llm_warning']
        assert len(r.json()['charts']) == 2


def test_plotly_svg_not_restricted_to_icon_dimensions():
    """Regression for heatmaps being invisible due to a global svg CSS selector."""
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / 'static/style.css').read_text(encoding='utf-8')
    assert 'svg{width:18px' not in css
    assert 'svg:has(> use)' in css


def _mock_helmcode(monkeypatch, response_handler):
    """Intercept Helmcode requests without external network or API credentials."""
    import httpx
    from sidechallenge import service

    native_client = httpx.AsyncClient
    transport = httpx.MockTransport(response_handler)
    monkeypatch.setattr(service.httpx, 'AsyncClient',
                        lambda **kwargs: native_client(transport=transport, **kwargs))
    monkeypatch.setenv('HELMCODE_API_KEY', 'test-secret-not-a-real-key')
    monkeypatch.setenv('HELMCODE_MODEL', 'qwen3.6')


def test_qwen_llm_json_mode_and_no_reasoning(monkeypatch):
    import json
    import httpx
    incoming = []

    def handler(request):
        outgoing = json.loads(request.content)
        incoming.append(outgoing)
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop','message': {
            'reasoning_content': 'private thinking is not the chart plan',
            'content':'{"kind":"heatmap","metrics":["precio_eur_per_kwh","co2_kg_per_kwh"],"group":"weekday","aggregation":"mean"}'
        }}]})

    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        response = client.post('/api/assistant',json={'message':'Heatmap de preu i CO2 per hora i dia de la setmana'})
        assert response.status_code == 200
        result = response.json()
        assert result['mode'] == 'helmcode'
        assert result['llm_warning'] is None
        assert len(result['charts']) == 2
        assert result['plan']['kind'] == 'heatmap'
    assert len(incoming) == 1
    assert incoming[0]['model'] == 'qwen3.6'
    assert incoming[0]['reasoning_effort'] == 'none'
    assert incoming[0]['max_tokens'] >= 1000
    assert incoming[0]['response_format'] == {'type':'json_object'}


def test_json_decoder_accepts_fenced_and_prefixed_responses():
    from sidechallenge.service import _extract_json_object
    expected = {'kind':'line','metrics':['load_kw'], 'group':'hour','aggregation':'mean'}
    import json
    assert _extract_json_object('```json\n'+json.dumps(expected)+'\n```') == expected
    assert _extract_json_object('Chart plan:\n'+json.dumps(expected)+'\nThanks!') == expected
    assert _extract_json_object([{'type':'text','text':json.dumps(expected)}]) == expected
    assert _extract_json_object(None) is None
    assert _extract_json_object('Reasoning only, no actual object') is None


def test_empty_final_reasoning_content_reported(monkeypatch):
    import httpx
    def handler(request):
        return httpx.Response(200, json={'choices':[{'finish_reason':'length',
            'message':{'reasoning_content':'the analysis used up the response budget','content':None}}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        response = client.post('/api/assistant', json={'message':'Heatmap preu CO2'})
        result = response.json()
        assert response.status_code == 200
        assert result['mode'] == 'local'
        assert 'tokens' in result['llm_warning']
        assert len(result['charts']) == 2


def test_json_mode_unsupported_retries_once(monkeypatch):
    import json
    import httpx
    outgoing = []
    def handler(request):
        body=json.loads(request.content)
        outgoing.append(body)
        if 'response_format' in body:
            return httpx.Response(400,json={'error':'unsupported response_format'})
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop',
            'message':{'content':'```json\n{"kind":"line","metrics":["load_kw"],"group":"hour","aggregation":"mean"}\n```'}}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        response = client.post('/api/assistant', json={'message':'Mostra el consum al llarg del dia'})
        assert response.status_code == 200
        assert response.json()['mode'] == 'helmcode'
    assert len(outgoing)==2
    assert 'response_format' not in outgoing[-1]


def test_malformed_model_fields_never_executed(monkeypatch):
    import httpx
    def handler(request):
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
            'content':'{"kind":"heatmap","metrics":["__class__"],"group":"hour","aggregation":"mean"}'}}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        response = client.post('/api/assistant', json={'message':'Fes heatmap preu CO2'})
        assert response.status_code == 200
        assert response.json()['mode'] == 'local'
        assert 'no admesos' in response.json()['llm_warning']


def test_helmcode_schema_variants_are_normalized(monkeypatch):
    import httpx
    def handler(request):
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{
            'content': '{"chart_type":"heatmap","variables":["price","co2"],"group_by":"day_of_week","agg":"average","title":"Gràfics"}'
        }}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        result=client.post('/api/assistant',json={'message':'Heatmap de preu i co2'}).json()
        assert result['mode'] == 'helmcode'
        assert result['llm_warning'] is None
        assert result['plan'] == {'dataset_id':'campus','kind':'heatmap',
                                  'metrics':['precio_eur_per_kwh','co2_kg_per_kwh'],
                                  'group':'weekday','aggregation':'mean'}
        assert len(result['charts']) == 2


def test_helmcode_multi_chart_response_combined(monkeypatch):
    import httpx
    def handler(request):
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
            'content': '{"charts":[{"type":"heatmap","metric":"precio_eur_per_kwh"},{"type":"heatmap","metric":"co2_kg_per_kwh"}]}'
        }}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        result=client.post('/api/assistant',json={'message':'Fes dos heatmaps de preu i co2'}).json()
        assert result['mode'] == 'helmcode'
        assert len(result['charts']) == 2


def test_helmcode_automatic_repair_on_unknown_metric(monkeypatch):
    import httpx
    import json
    outgoing=[]
    def handler(request):
        outgoing.append(json.loads(request.content))
        if len(outgoing)==1:
            content='{"chart_type":"heatmap","variables":["preu_mitja","co2_mitja"]}'
        else:
            content='{"kind":"heatmap","metrics":["precio_eur_per_kwh","co2_kg_per_kwh"],"group":"weekday","aggregation":"mean"}'
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':content}}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        result=client.post('/api/assistant',json={'message':'Dos heatmaps de preu i co2'}).json()
        assert result['mode'] == 'helmcode'
        assert len(result['charts']) == 2
        assert result['llm_warning'] is None
    assert len(outgoing)==2
    assert 'validation_error' in outgoing[1]['messages'][1]['content']


def test_llm_valid_but_incomplete_price_only_is_completed(monkeypatch):
    """Regression: a valid JSON containing only price must not drop requested CO2."""
    import httpx
    def handler(request):
        return httpx.Response(200, json={"choices":[{"finish_reason":"stop", "message":{
            "content":"{\"kind\":\"heatmap\",\"metrics\":[\"precio_eur_per_kwh\"],\"group\":\"weekday\",\"aggregation\":\"mean\"}"
        }}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        result = client.post('/api/assistant', json={
            'message': 'Fes dos heatmaps, un dels preus i l’altre de les emissions de CO₂, per hora i dia de la setmana'
        }).json()
    assert result['mode'] == 'helmcode'
    assert result['llm_warning'] is None
    assert result['plan']['metrics'] == ['precio_eur_per_kwh','co2_kg_per_kwh']
    assert result['plan']['kind'] == 'heatmap'
    assert len(result['charts']) == 2
    assert len(result['intent_adjustments']) == 1
    assert 'kg CO₂/kWh' in ' '.join(result['insights'])


def test_llm_valid_but_wrong_chart_type_is_reconciled(monkeypatch):
    import httpx
    def handler(request):
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop', 'message':{
            'content':'{"kind":"line","metrics":["precio_eur_per_kwh","co2_kg_per_kwh"],"group":"hour","aggregation":"mean"}'
        }}]})
    _mock_helmcode(monkeypatch, handler)
    with TestClient(app) as client:
        result = client.post('/api/assistant', json={
            'message':'Vull un mapa de calor de preus i emissions per dia de la setmana'
        }).json()
    assert result['mode'] == 'helmcode'
    assert result['plan']['kind'] == 'heatmap'
    assert result['plan']['group'] == 'weekday'
    assert len(result['charts']) == 2


def test_metric_requirement_unicode_subscript_and_availability():
    from sidechallenge.service import _query_requirements, _complete_user_intent
    fields=['precio_eur_per_kwh','co2_kg_per_kwh', 'load_kw']
    selected, heatmap = _query_requirements('Heatmaps de preu i CO₂', fields)
    assert selected == ['precio_eur_per_kwh','co2_kg_per_kwh'] and heatmap
    selected, _ = _query_requirements('Heatmap del preu', ['precio_eur_per_kwh'])
    assert selected == ['precio_eur_per_kwh']
    plan, changes = _complete_user_intent({'kind':'heatmap','metrics':['precio_eur_per_kwh'], 'group':'weekday','aggregation':'mean'}, 'Heatmaps del preu i les emissions', ['precio_eur_per_kwh'], True)
    assert plan['metrics'] == ['precio_eur_per_kwh'] and not changes

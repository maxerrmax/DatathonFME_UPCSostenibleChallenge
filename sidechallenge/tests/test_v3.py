"""v3: official physics, Agent Arena, robust EDA and the tool-using analyst."""
import json
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sidechallenge import arena, copilot, eda, service

DF = service.DATASETS["campus"].frame


def test_official_eval_matches_environment_exactly():
    Env = arena._env_class()
    week = arena.pseudo_test_weeks(DF)[0]["frame"]
    actions = np.random.default_rng(1).uniform(-800, 800, len(week))
    env = Env(week[["Fecha", *arena.COLS]].set_index("Fecha")); env.reset()
    for a in actions:
        env.step(float(a))
    official = env.evaluate_results()
    replica = arena.official_eval(week, actions)
    assert replica["score"] == official["score"]
    assert replica["infractions"] == official["infraccions"]
    assert abs(replica["co2_saved_kg"] - official["co2_estalviat_kg"]) < 0.1


def test_lp_schedules_never_trigger_infractions():
    weeks = arena.pseudo_test_weeks(DF)
    assert len(weeks) >= 10
    for w in weeks[:4]:
        f = w["frame"]
        plan = arena.lp_schedule(f["load_kw"], f["pv_kw"], f["precio_eur_per_kwh"], f["co2_kg_per_kwh"], 1000.0)
        assert arena.run_official(arena.ScheduleAgent(plan, "o"), f)["infractions"] == 0


def test_what_if_parser_handles_order_language_and_ranges():
    p = copilot.parse_what_if("Descarrego a les 20 i carrego a les 3")
    assert p["charge"] == (3, 3) and p["discharge"] == (20, 20)
    p = copilot.parse_what_if("What if I discharge at 19 and charge at 2?")
    assert p["charge"] == (2, 2) and p["discharge"] == (19, 19)
    p = copilot.parse_what_if("Què passaria si carrego entre les 13 i les 16, i descarrego de 20 a 23, a 250 kW?")
    assert p["charge"] == (13, 16) and p["discharge"] == (20, 23) and p["power"] == 250


def test_short_new_question_is_not_hijacked_by_context():
    with TestClient(service.app) as c:
        r = c.post("/api/assistant", json={"message": "Quines hores són les més netes i barates?",
                                           "previous_plan": {"kind": "heatmap", "metrics": ["load_kw"], "group": "weekday", "aggregation": "mean"}}).json()
    assert r["mode"] != "context"
    assert r["intent"] == "best_hours"
    assert r["verification"]["ok"]


def test_mixed_units_never_share_an_axis():
    req = service.ChartReq(kind="line", metrics=["precio_eur_per_kwh", "co2_kg_per_kwh", "load_kw"], group="hour")
    fig = service.make_charts(req)[0]["figure"]
    axes = {t.get("yaxis", "y") for t in fig["data"] if t.get("name")}
    assert len(axes) == 3
    hist = service.make_charts(service.ChartReq(kind="histogram", metrics=["precio_eur_per_kwh", "load_kw"]))[0]["figure"]
    assert {t.get("xaxis", "x") for t in hist["data"]} == {"x", "x2"}


def test_daytype_comparison_uses_means_with_intervals():
    with TestClient(service.app) as c:
        d = c.post("/api/assistant", json={"message": "Ara compara-ho amb els caps de setmana",
                                           "previous_plan": {"kind": "line", "metrics": ["load_kw"], "group": "hour", "aggregation": "sum"}}).json()
    assert d["comparison"] and "mitjana" in " ".join(d["insights"])
    trace = d["charts"][0]["figure"]["data"][0]
    assert "error_y" in trace and len(trace["x"]) == 24


def test_quality_overview_and_radar_are_consistent():
    with TestClient(service.app) as c:
        q = c.get("/api/quality").json()
        assert any("congelades" in i["title"] for i in q["issues"])
        assert 11 in q["months_absent"]
        o = c.get("/api/overview").json()
        assert 11 not in o["dataset"]["months"] and o["frozen_hours"]
        assert len(o["robust"]["load_kw"]["median"]) == 24
        radar = c.get("/api/radar").json()
        assert radar["candidate"]["days_examined"] == len(c.get("/api/days").json()["dates"])
        assert len(radar["by_month"]) >= 10


def test_lab_windows_closed_cycle_and_capture():
    day = arena.complete_days(DF)[-1]
    with TestClient(service.app) as c:
        x = c.post("/api/simulate", json={"date": day, "charge_start": 13, "charge_end": 16,
                                          "discharge_start": 20, "discharge_end": 23}).json()
    assert x["manual"]["infractions"] == 0 and x["optimal"]["infractions"] == 0
    assert min(x["manual_soc"]) >= 1000 - 1e-6
    assert 0 < x["capture_pct"] <= 100.5


def test_arena_runs_official_backtest():
    df = DF
    arena.start_background(df)
    for _ in range(240):
        if arena.ARENA.snapshot()["status"] in ("ready", "error"):
            break
        time.sleep(0.5)
    snap = arena.ARENA.snapshot()
    assert snap["status"] == "ready", snap
    res = arena.ARENA.result
    keys = {p["key"] for p in res["policies"]}
    assert {"idle", "rule", "team", "mpc_naive", "mpc_perfect", "oracle"} <= keys
    score = {p["key"]: p["pooled"]["score"] for p in res["policies"]}
    assert score["oracle"] >= score["rule"] > score["idle"] == 0
    assert all(p["pooled"]["infractions"] == 0 for p in res["policies"] if p["key"] != "team")


def test_clean_args_rejects_unknown_fields_and_injection():
    fields = ["load_kw", "precio_eur_per_kwh"]
    assert copilot.clean_args("hourly_profile", {"metric": "__class__", "month": 99, "evil": 1}, fields) == {"month": 12}
    assert copilot.clean_args("simulate_day", {"date": "../../etc", "power": 9000}, fields) == {"power": 500.0}


def test_verifier_flags_invented_numbers():
    ev = [{"summary": {"score": 6.4512, "pct": 5.968}, "facts": ["Marge 0,664"]}]
    assert copilot.verify("Score 6,45 i CO₂ 5,97% [E1], marge 0,66.", ev)["ok"]
    bad = copilot.verify("Score 6,45 però estalvia 12,8% [E1].", ev)
    assert not bad["ok"] and "12,8" in bad["unverified"]


def _mock(monkeypatch, handler):
    native = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(copilot.httpx, "AsyncClient", lambda **kw: native(transport=transport, **kw))
    monkeypatch.setenv("HELMCODE_API_KEY", "test-secret-not-a-real-key")
    monkeypatch.setenv("HELMCODE_MODEL", "qwen3.6")


def test_llm_tool_calling_loop_is_grounded(monkeypatch):
    seen = []

    def handler(request):
        body = json.loads(request.content); seen.append(body)
        if len(seen) == 1:
            assert any(t["function"]["name"] == "best_hours" for t in body["tools"])
            return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "best_hours", "arguments": "{\"month\": 7}"}}]}}]})
        tool_msg = next(m for m in body["messages"] if m["role"] == "tool")
        payload = json.loads(tool_msg["content"])
        c, d = payload["summary"]["charge_hour"], payload["summary"]["discharge_hour"]
        return httpx.Response(200, json={"choices": [{"message": {"content":
            f"Al juliol convé carregar a les {c} h i descarregar a les {d} h [E1]. També estalviaria un 37,5% [E1].\nDecisió: provar-ho al Lab."}}]})

    _mock(monkeypatch, handler)
    with TestClient(service.app) as client:
        r = client.post("/api/assistant", json={"message": "Quines hores són millors per carregar al juliol?"}).json()
    assert r["mode"] == "analyst-helmcode"
    assert r["trace"][0]["tool"] == "best_hours" and r["trace"][0]["args"] == {"month": 7}
    assert seen[0]["reasoning_effort"] == "none"
    assert not r["verification"]["ok"] and "37,5" in r["verification"]["unverified"]


def test_tools_unsupported_falls_back_to_json_plan(monkeypatch):
    calls = []

    def handler(request):
        body = json.loads(request.content); calls.append(body)
        if "tools" in body:
            return httpx.Response(400, json={"error": "tools are not supported"})
        if body.get("response_format"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "{\"steps\":[{\"tool\":\"data_quality\",\"args\":{}}]}"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi ha lectures congelades [E1].\nDecisió: excloure-les."}}]})

    _mock(monkeypatch, handler)
    with TestClient(service.app) as client:
        r = client.post("/api/assistant", json={"message": "Les dades són fiables?"}).json()
    assert r["mode"] == "analyst-helmcode-json" and r["trace"][0]["tool"] == "data_quality"


def test_provider_failure_uses_local_analyst(monkeypatch):
    _mock(monkeypatch, lambda request: httpx.Response(503, json={"error": "down"}))
    with TestClient(service.app) as client:
        r = client.post("/api/assistant", json={"message": "Quina estratègia de bateria funciona millor?"}).json()
    assert r["mode"] == "analyst-local" and "Helmcode" in r["llm_warning"]
    assert r["trace"][0]["tool"] == "backtest"

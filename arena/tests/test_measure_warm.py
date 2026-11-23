"""Opt-in warm-up of the ms/node measurement (--measure-warm-ms, docs/speed_gap.md): off by default, passed through
to every wave 2 / upstream arm when set, and the warm searches are not counted."""

from arena.players import upstream as up
from arena.players.hivemind import SearchResult


class _FakeProcess:
    """Stands in for UpstreamHivemindProcess: each search takes ``ms`` of wall time, the first ``cold`` 800-node
    searches ``slow_ms``."""

    def __init__(self, cold=0, ms=36, slow_ms=60, **kw):
        self.calls, self.cold, self.ms, self.slow_ms = [], cold, ms, slow_ms

    def search_clock(self, fa, fb, *, nodes, **kw):
        self.calls.append(nodes)
        big = [n for n in self.calls if n > 64]
        wall = self.slow_ms if nodes > 64 and len(big) <= self.cold else self.ms
        return SearchResult(move_a="a2a3", move_b=None, q=0.0, cp=0, mate=None, nodes=nodes, depth=1, wall_ms=wall)

    def close(self):
        pass


def test_default_measurement_is_unchanged(monkeypatch):
    made = []
    monkeypatch.setattr(up, "UpstreamHivemindProcess", lambda **kw: made.append(_FakeProcess(cold=1)) or made[-1])
    ms = up.measure_upstream_ms_per_node(nodes=800, repeats=3)
    assert made[0].calls == [64, 800, 800, 800]  # the 64-node warm-up, then the three measured searches
    assert abs(ms - (60 + 36 + 36) / 2400) < 1e-12  # the cold first search is counted, as before


def test_warm_ms_searches_first_and_does_not_count_them(monkeypatch):
    made = []
    monkeypatch.setattr(up, "UpstreamHivemindProcess", lambda **kw: made.append(_FakeProcess(cold=1)) or made[-1])
    clock = iter(range(0, 10_000, 1))  # perf_counter in s: one second per call
    monkeypatch.setattr(up.time, "perf_counter", lambda: next(clock))
    ms = up.measure_upstream_ms_per_node(nodes=800, repeats=3, warm_ms=1500)
    assert made[0].calls == [64, 800, 800, 800, 800, 800]  # two warm searches (1 s each by the fake clock), then three
    assert abs(ms - 36 / 800) < 1e-12


def test_apply_passes_warm_ms_only_when_set(tmp_path):
    from arena.pool import Pool, apply_ms_per_node, for_evaluation

    pool = Pool.load("arena/pools/wave2.json")
    row = for_evaluation(pool, opponent="upstream-it13", contender_kw={"ms_per_node": "measure"},
                         opponent_kw={"ms_per_node": "measure"})
    seen = []

    def fake(b, m, nodes, repeats, coreml=None, wave2=None, **kw):
        seen.append((wave2 is not None, repeats, kw))
        return {"ms_per_node": 0.05}

    apply_ms_per_node(row, str(tmp_path / "a.json"), measure_upstream=fake)
    assert sorted(seen) == [(False, 3, {}), (True, 3, {})]
    seen.clear()
    apply_ms_per_node(row, str(tmp_path / "b.json"), measure_upstream=fake, warm_ms=2000, repeats=10)
    assert sorted(seen) == [(False, 10, {"warm_ms": 2000}), (True, 10, {"warm_ms": 2000})]


def test_measure_upstream_arm_records_warm_ms(monkeypatch):
    import arena.pool as pool_mod

    got = {}

    def fake_measure(nodes, repeats, **kw):
        got.update(kw)
        return 0.043

    monkeypatch.setattr(up, "measure_upstream_ms_per_node", fake_measure)
    rec = pool_mod.measure_upstream_arm("bin", "model.onnx", nodes=800, repeats=10, warm_ms=2000)
    assert got["warm_ms"] == 2000 and rec["warm_ms"] == 2000 and rec["ms_per_node"] == 0.043
    got.clear()
    rec = pool_mod.measure_upstream_arm("bin", "model.onnx", nodes=800, repeats=3)
    assert "warm_ms" not in got and "warm_ms" not in rec


def test_cached_record_with_other_warm_ms_is_remeasured(tmp_path):
    import json

    from arena.pool import Pool, apply_ms_per_node, for_evaluation

    pool = Pool.load("arena/pools/wave2.json")
    row = for_evaluation(pool, opponent="upstream-it13", contender_kw={"ms_per_node": "measure"},
                         opponent_kw={"ms_per_node": "measure"})
    calls = []

    def fake(b, m, nodes, repeats, coreml=None, wave2=None, warm_ms=0, **kw):
        calls.append(warm_ms)
        return {"ms_per_node": 0.05 if warm_ms == 0 else 0.043, **({"warm_ms": warm_ms} if warm_ms else {})}

    cache = str(tmp_path / "c.json")
    apply_ms_per_node(row, cache, measure_upstream=fake)  # cold records
    assert calls == [0, 0]
    calls.clear()
    apply_ms_per_node(row, cache, measure_upstream=fake)  # same procedure: reused
    assert calls == []
    new, used = apply_ms_per_node(row, cache, measure_upstream=fake, warm_ms=3000)  # warm asked: re-measured
    assert calls == [3000, 3000]
    assert all(r["warm_ms"] == 3000 and r["ms_per_node"] == 0.043 for r in used.values())
    assert all(r.get("warm_ms") == 3000 for r in json.load(open(cache)).values())
    calls.clear()
    apply_ms_per_node(row, cache, measure_upstream=fake, warm_ms=3000)  # resumed warm row: reused
    assert calls == []
    apply_ms_per_node(row, cache, measure_upstream=fake)  # cold asked of a warm cache: re-measured
    assert calls == [0, 0]
    calls.clear()
    apply_ms_per_node(row, cache, measure_upstream=fake, warm_ms=3000,
                      override={k: 0.04 for k in used})  # overrides are never re-measured
    assert calls == []

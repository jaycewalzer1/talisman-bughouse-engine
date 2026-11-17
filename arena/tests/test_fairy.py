"""Fairy-Stockfish CECP arm (arena/players/fairy.py): commands sent (holdings, time/otim in centiseconds, partner
clocks), ptell handling (dead -> sit, sit -> partner sits), and the real engine when it is built."""

from __future__ import annotations

import pytest

from arena.interfaces import Message, Observation
from arena.players.fairy import FairyPlayer, FairyProcess, FairyResult, fairy_available

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/6K1[QQ] w - - 0 30"  # Q@a8 / Q@e8 mates: needs the pocket
MATED = "6k1/5ppp/8/8/8/8/5PPP/6K1[qq] w - - 0 30"  # Black drops a queen and mates whatever White does


class Recorder(FairyProcess):
    """FairyProcess with the subprocess replaced by a script of output lines."""

    def __init__(self, script):
        super().__init__("/nonexistent")
        self.sent: list[str] = []
        self.script = list(script)

    def start(self):
        pass

    def _send(self, cmd):
        self.sent.append(cmd)
        if cmd.startswith("ping"):
            self._q.put(f"pong {cmd.split()[1]}")
        if cmd == "go":
            for line in self.script:
                self._q.put(line)


def _obs(seat, fen, clocks, legal, inbox=()):
    return Observation(now_ms=0, seat=seat, fens={"A": fen, "B": fen}, clocks_ms=clocks,
                       to_move={"A": "A_white", "B": "B_white"}, legal_moves=legal, inbox=tuple(inbox))


CLOCKS = {"A_white": 41_230, "A_black": 20_000, "B_white": 50_000, "B_black": 30_000}


def test_commands_carry_holdings_and_clocks_in_centiseconds():
    proc = Recorder([" 10 25 12 120000 14 1000000 0\tQ@a8", "move Q@a8"])
    p = FairyPlayer("A_white", proc, base_ms=60_000)
    dec = p.decide(_obs("A_white", MATE_IN_ONE, CLOCKS, ["Q@a8", "g1f1"]))
    sent = proc.sent
    assert f"setboard {MATE_IN_ONE}" in sent  # the bracket FEN carries the holdings
    assert "level 0 1 0" in sent
    assert "time 4123" in sent and "otim 2000" in sent  # our clock, our opponent's (A_black), in cs
    assert "ptell time 3000" in sent and "ptell otim 5000" in sent  # partner B_black and its opponent B_white
    assert sent.index("go") > sent.index("otim 2000")
    assert dec.kind == "move" and dec.uci == "Q@a8" and dec.nodes == 120000


def test_dead_ptell_makes_the_seat_sit_and_tells_partner():
    proc = Recorder([" 9 -100005 1 6000 9 600000 0\t", "tellics ptell dead", "move g2g3"])
    p = FairyPlayer("A_white", proc, base_ms=60_000, sit_ms=700)
    dec = p.decide(_obs("A_white", MATED, CLOCKS, ["g2g3", "h2h3"]))
    assert dec.kind == "wait" and dec.stall_reason == "fairy_dead" and dec.wait_ms == dec.think_ms + 700
    assert [m.token for m in dec.messages] == ["fairy_dead"] and dec.messages[0].to_seat == "B_black"
    assert "ptell dead" in proc.sent  # sent back at once so Fairy's own wall-clock sit loop ends


def test_partner_sit_and_x():
    proc = Recorder(["move e2e4"])
    p = FairyPlayer("B_black", proc)
    sit = Message("A_white", "B_black", "fairy_sit", 0, payload={"ptell": "sit"})
    clocks = dict(CLOCKS)
    dec = p.decide(_obs("B_black", MATE_IN_ONE, clocks, ["e2e4"], inbox=[sit]))
    assert dec.kind == "wait" and dec.stall_reason == "partner_sit" and not proc.sent  # no search while sitting
    x = Message("A_white", "B_black", "fairy_x", 0, payload={"ptell": "x"})
    assert p.decide(_obs("B_black", MATE_IN_ONE, clocks, ["e2e4"], inbox=[x])).kind == "move"
    p2 = FairyPlayer("B_black", Recorder(["move e2e4"]))
    low = dict(CLOCKS, B_black=900)  # under 1 s Fairy stops sitting
    assert p2.decide(_obs("B_black", MATE_IN_ONE, low, ["e2e4"], inbox=[sit])).kind == "move"


def test_result_parsing_virtual_drop_wait():
    proc = Recorder([" 12 300 50 400000 15 2000000 0\tQ@f7", "tellics ptell fast", "tellics ptell x", "move e2e4"])
    r = proc.think(MATE_IN_ONE, base_ms=60_000, own_ms=60_000, opp_ms=60_000, partner_ms=60_000,
                   partner_opp_ms=10_000)
    assert isinstance(r, FairyResult) and r.wants_piece and r.move == "e2e4" and r.depth == 12
    assert "ptell fast" in proc.sent


needs_fairy = pytest.mark.skipif(not fairy_available(), reason="build Fairy-Stockfish (NOTES.md)")


@needs_fairy
def test_engine_uses_holdings_and_otim():
    proc = FairyProcess()
    try:
        r = proc.think(MATE_IN_ONE, base_ms=60_000, own_ms=60_000, opp_ms=20_000, partner_ms=30_000,
                       partner_opp_ms=50_000)
        assert r.move in ("Q@a8", "Q@b8", "Q@c8", "Q@d8", "Q@e8") and r.score >= 100000
        assert "sit" in [w.split()[0] for w in r.ptells]  # mate found and the opponent is shorter than partner
        r2 = proc.think(MATE_IN_ONE, base_ms=60_000, own_ms=60_000, opp_ms=40_000, partner_ms=30_000,
                        partner_opp_ms=50_000)
        assert "sit" not in [w.split()[0] for w in r2.ptells]  # otim above partner's time: no sit request
        empty = proc.think(MATE_IN_ONE.replace("[QQ]", "[]"), base_ms=60_000, own_ms=60_000, opp_ms=40_000,
                           partner_ms=30_000, partner_opp_ms=50_000)
        assert "@" not in (empty.move or "")  # no holdings, no drops
    finally:
        proc.close()


def test_fixed_move_time_sends_st_after_clocks_and_charges_it():
    proc = Recorder([" 10 25 12 120000 14 1000000 0\tQ@a8", "move Q@a8"])
    p = FairyPlayer("A_white", proc, base_ms=300_000, charge="move", move_s=2)
    dec = p.decide(_obs("A_white", MATE_IN_ONE, CLOCKS, ["Q@a8", "g1f1"]))
    sent = proc.sent
    assert "st 2" in sent and sent.index("st 2") > sent.index("otim 2000") and sent.index("st 2") < sent.index("go")
    assert dec.kind == "move" and dec.think_ms == 2000
    # default: no st
    proc = Recorder([" 10 25 12 120000 14 1000000 0\tQ@a8", "move Q@a8"])
    FairyPlayer("A_white", proc, base_ms=60_000).decide(_obs("A_white", MATE_IN_ONE, CLOCKS, ["Q@a8", "g1f1"]))
    assert not any(c.startswith("st ") for c in proc.sent)
    with pytest.raises(ValueError):
        FairyPlayer("A_white", Recorder([]), charge="move")

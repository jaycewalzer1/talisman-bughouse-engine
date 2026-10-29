"""Live WebSocket server for the viewers.

    python -m arena.serve --port 8765 --kind heuristic --contender sharing --baseline silent \
        --speed 1.0 --base-ms 60000
    python -m arena.serve --port 8765 --replay runs/x/game_0001.jsonl

Plays games back to back (contender/baseline teams swapped each game, openings from
arena/openings.py) and broadcasts each event as one JSON text frame, paced so that an event with
game time `ts` goes out `ts / speed` ms after the game's first event. The simulator runs in a worker
thread and events are forwarded as they are produced; if the simulator is slower than real time,
events go out as soon as they exist and pacing resumes from there.

A client that connects mid-game first receives every event of the current game so far (immediately),
then the live ones. Between games the server pauses (--pause-s); during the pause new clients receive
the finished game. Any Origin is accepted.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import threading
import time

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from arena.openings import paired_schedule

log = logging.getLogger("arena.serve")
_DONE = object()


class Hub:
    """Holds the current game's events and fans them out. All methods run on the event loop thread
    and never await, so a joining client's snapshot and the live stream cannot interleave."""

    def __init__(self) -> None:
        self.current: list[str] = []  # serialized events of the current game
        self.clients: set[asyncio.Queue] = set()

    def publish(self, event: dict) -> None:
        if event.get("type") == "game_start":
            self.current = []
        frame = json.dumps(event, separators=(",", ":"))
        self.current.append(frame)
        for q in self.clients:
            q.put_nowait(frame)

    def join(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        for frame in self.current:
            q.put_nowait(frame)
        self.clients.add(q)
        return q

    def leave(self, q: asyncio.Queue) -> None:
        self.clients.discard(q)


async def handle(ws: ServerConnection, hub: Hub) -> None:
    peer = ws.remote_address
    q = hub.join()
    log.info("client connected %s (%d clients, %d events of current game queued)", peer, len(hub.clients),
             q.qsize())
    try:
        while True:
            await ws.send(await q.get())
    except ConnectionClosed:
        pass
    finally:
        hub.leave(q)
        log.info("client disconnected %s (%d clients)", peer, len(hub.clients))


class Pacer:
    """Sleeps until wall time t0 + ts/speed; if we're already late, shifts t0 so we never rush."""

    def __init__(self, speed: float) -> None:
        self.speed = speed
        self.t0: float | None = None

    async def wait(self, ts_ms: int) -> None:
        now = time.monotonic()
        offset = ts_ms / 1000.0 / self.speed
        if self.t0 is None:
            self.t0 = now - offset
        delay = self.t0 + offset - now
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            self.t0 = now - offset


async def broadcast_paced(source: asyncio.Queue, hub: Hub, speed: float) -> None:
    """Drain `source` (events, then _DONE) into the hub, pacing by event ts."""
    pacer = Pacer(speed)
    while True:
        ev = await source.get()
        if ev is _DONE:
            return
        await pacer.wait(int(ev.get("ts", 0)))
        hub.publish(ev)


def _start_sim_thread(spec: dict, loop: asyncio.AbstractEventLoop, out: asyncio.Queue) -> threading.Thread:
    """Run one game in a thread, forwarding each event to `out` as it is produced."""
    from arena.run import event_to_dict, play_game

    def push(ev) -> None:
        loop.call_soon_threadsafe(out.put_nowait, ev)

    def target() -> None:
        seen: set = set()

        def on_event(e) -> None:
            ev = event_to_dict(e)
            seen.add(ev.get("seq"))
            push(ev)

        try:
            events, result = play_game(spec, on_event=on_event)
            # in case the simulator does not call on_event for every event
            for ev in events:
                if ev.get("seq") not in seen:
                    push(ev)
            log.info("game %s finished: %s", spec["game_id"], result)
        except Exception:  # noqa: BLE001
            log.exception("game %s failed", spec["game_id"])
        finally:
            push(_DONE)

    t = threading.Thread(target=target, name=f"sim-{spec['game_id']}", daemon=True)
    t.start()
    return t


async def live_loop(args, hub: Hub) -> None:
    loop = asyncio.get_running_loop()
    schedule = paired_schedule(10**9, args.seed)
    for i in itertools.count():
        pair, swap = next(schedule)
        t1, t2 = (args.baseline, args.contender) if swap else (args.contender, args.baseline)
        spec = {"game_id": f"live{i + 1:04d}", "t1_setup": t1, "t2_setup": t2, "kind": args.kind,
                "seed": args.seed * 100003 + i, "nodes": args.nodes, "base_ms": args.base_ms,
                "increment_ms": args.increment_ms, "fens": pair.fens,
                "channel_delay_ms": args.channel_delay_ms, "max_plies": args.max_plies,
                "charge_wall_time": args.charge_wall_time,
                "player_kw": {"compute": args.compute, "lend": args.lend}}
        log.info("game %s: %s, T1=%s T2=%s (contender %s)", spec["game_id"], pair.name, t1, t2,
                 "T2" if swap else "T1")
        q: asyncio.Queue = asyncio.Queue()
        _start_sim_thread(spec, loop, q)
        await broadcast_paced(q, hub, args.speed)
        await asyncio.sleep(args.pause_s)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def replay_loop(args, hub: Hub) -> None:
    events = read_jsonl(args.replay)
    log.info("replaying %s (%d events)", args.replay, len(events))
    while True:
        q: asyncio.Queue = asyncio.Queue()
        for ev in events:
            q.put_nowait(ev)
        q.put_nowait(_DONE)
        await broadcast_paced(q, hub, args.speed)
        if args.once:
            return
        await asyncio.sleep(args.pause_s)


async def main_async(args) -> None:
    hub = Hub()
    async with serve(lambda ws: handle(ws, hub), args.host, args.port, origins=None) as server:
        ports = sorted({s.getsockname()[1] for s in server.sockets})
        log.info("serving on ws://%s:%s (%s)", args.host, ",".join(map(str, ports)),
                 f"replay {args.replay}" if args.replay else
                 f"{args.kind}: {args.contender} vs {args.baseline}, speed {args.speed}")
        producer = replay_loop(args, hub) if args.replay else live_loop(args, hub)
        await producer
        await server.serve_forever()  # --once: keep serving the finished game


def parse_args(argv=None):
    from arena.run import KINDS, SETUPS

    ap = argparse.ArgumentParser(prog="python -m arena.serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--kind", default="heuristic", choices=KINDS)
    ap.add_argument("--contender", default="sharing", choices=SETUPS)
    ap.add_argument("--baseline", default="silent", choices=SETUPS)
    ap.add_argument("--speed", type=float, default=1.0, help="game-time ms per wall-time ms")
    ap.add_argument("--base-ms", type=int, default=60000)
    ap.add_argument("--increment-ms", type=int, default=0)
    ap.add_argument("--channel-delay-ms", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--compute", default="throughput", choices=("throughput", "nodes"))
    ap.add_argument("--lend", action="store_true", help="hivemind split setups: borrow the partner's idle machine")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--charge-wall-time", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause-s", type=float, default=5.0, help="pause between games")
    ap.add_argument("--replay", default=None, help="stream a recorded game_XXXX.jsonl (looped)")
    ap.add_argument("--once", action="store_true", help="with --replay: play the file once, then idle")
    args = ap.parse_args(argv)
    if args.speed <= 0:
        ap.error("--speed must be > 0")
    return args


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    args = parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

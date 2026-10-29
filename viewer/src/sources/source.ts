// The tiny interface every event source (mock, live WebSocket, replay file) implements.
// The controller (./controller.ts) owns exactly one active source at a time.

export interface Source {
  /** Begin producing events (open socket, generate first mock game, ...). */
  start(): void
  /** Stop everything (close socket, cancel timers). The source is not reused after this. */
  stop(): void
  /**
   * Called on every animation frame.
   * @param dtMs   simulated milliseconds since the last frame (wall time x speed; 0 when paused)
   * @param wallNow performance.now(), for sources that follow real time (live)
   */
  tick(dtMs: number, wallNow: number): void
}

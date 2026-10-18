# Supervised encoding (Python) disagrees with engine/self-play encoding (C++)

**Commit:** 1d01d6d

Cross-checking `generate_planes`/`board2planes` against `get_fast_policy_index`/`board_to_planes` on replayed
random bughouse games found five systematic differences. In each case the C++ side follows Fairy-Stockfish,
and the released `it04` network follows C++ for pass and O-O.

## Policy labels (100% of affected moves)
| Action | Python label → index | Engine index |
|---|---|---|
| pass | 63 (`{label: index}` over `make_map()` keeps the last of 64 duplicate `'pass'` slots) | 0 |
| O-O | `e1g1` → 1348 | 1412 (`e1h1`, king→rook) |
| O-O-O | `e1c1` → 3140 | 3268 (`e1a1`) |

Network check (logit rank among legal moves, 0 = best): pass@0 0.17 vs pass@63 1.00; O-O@e1h1 0.59 vs @e1g1 0.97.
O-O-O: in standard queenside-castling openings (Yugoslav, English Attack, Caro-Kann classical) the engine index
`e1a1` ranks 44–51 of 47–52 legal moves, and the old Python slot `e1c1` 33–42 — the network has effectively not
learned O-O-O at the index the engine reads, so the engine almost never castles long.

Side effect: `tests/data/test_generate_planes.py` looks for pass at `labels.index('pass')` (0), so its
both-boards-pass check could never fire on data labelled 63.

## Input planes (≈55% of replayed positions)
| Channel | python-chess | Engine (Fairy-Stockfish) |
|---|---|---|
| ep_square | set after every double push | only when ep capture is legal |
| last_to after castling | king square (g1/c1) | rook square (h1/a1) |
| halfmove_clock | not reset by pawn drops | reset by pawn drops |

End-to-end value gap up to 0.17; top-1 legal move differed in 18/998 board positions. After aligning Python to the
engine: 3005/3005 positions identical on all 74 channels, values agree to ~1e-6.

Patch available: <fork link> (commits f553bba, b44ee22), with tests in `tests/domain/test_policy_labels.py` and
`tests/domain/test_engine_plane_conventions.py`. Parity harnesses: <link to talisman tools>.

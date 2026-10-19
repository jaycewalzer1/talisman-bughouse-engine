# talisman-bughouse-engine

Research code for *Think, Talk, or Move: How AI Teammates Should Share Time and
Computation in Bughouse Chess*. Built on [Hivemind](https://github.com/aminwoo/hivemind)
and [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish).

```bash
git clone --recurse-submodules https://github.com/jaycewalzer1/talisman-bughouse-engine.git
```

Setup, baseline status and parity results are in [NOTES.md](NOTES.md).

- `third_party/hivemind`: fork of Hivemind, branch `talisman/baseline-fixes`
  (Clang `-Ofast` search fix, Python/engine encoding alignment).
- `third_party/Fairy-Stockfish`: upstream, unmodified.
- `tools/`: Python/C++ parity harnesses and network probes.
- `docker/`: Linux test image.

## License

Code in this repository is MIT licensed (see [LICENSE](LICENSE)). Submodules keep
their own licenses: Hivemind is MIT; Fairy-Stockfish is GPL-3.0. The helpers in
`tools/cpp` link against Fairy-Stockfish, so binaries built from them are covered
by the GPL-3.0.

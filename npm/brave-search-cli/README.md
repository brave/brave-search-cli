# @brave/brave-search-cli

`bx` — a token-efficient CLI for the [Brave Search API](https://brave.com/search/api/), built for AI agents and LLMs.

```bash
npm install -g @brave/brave-search-cli
bx config set-key YOUR_API_KEY
bx "your search query"
```

This package installs the prebuilt `bx` binary for your platform (via optional dependencies — no build step, no postinstall download, so it works under `--ignore-scripts`). Supported platforms: macOS (arm64, x64), Linux (x64, arm64), Windows (x64, arm64). Selection follows `process.arch`, not the CPU, so an x64 Node on Apple Silicon runs the Intel binary under Rosetta — use an arm64 build of Node for the native one.

This package launches `bx` through a small Node shim, which adds ~20 ms per invocation. If that matters, [install the binary directly](https://github.com/brave/brave-search-cli#quick-start).

Full documentation: https://github.com/brave/brave-search-cli

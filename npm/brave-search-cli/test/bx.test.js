'use strict'

// Black-box tests for the npm launcher. No network, no API key, no deps.
//
// Each case runs the real bin/bx.js from a throwaway tree shaped like a real
// install — <root>/bin/bx.js next to <root>/node_modules/<platform pkg>/bin/bx —
// so require.resolve walks the same path it would on a user's machine. The
// launcher source is copied byte-for-byte, never rewritten.

const assert = require('node:assert/strict')
const { spawnSync } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { after, before, beforeEach, test } = require('node:test')

const SRC = path.join(__dirname, '..', 'bin', 'bx.js')
const HOST_PKG = `@brave/brave-search-cli-${process.platform}-${process.arch}`
const HOST_BIN = process.platform === 'win32' ? 'bx.exe' : 'bx'

const linuxOnly = { skip: process.platform !== 'linux' && 'Linux-specific' }
const posixOnly = { skip: process.platform === 'win32' && 'POSIX shebangs/signals' }

// A Node stub round-trips argv byte-exactly. Signals need a /bin/sh stub: Node
// cannot re-raise one it has no name for, which is the case that matters here.
const NODE_STUB = `#!${process.execPath}
'use strict'
const fs = require('node:fs')
switch (process.env.STUB_MODE) {
  case 'exit':
    process.exit(Number(process.env.STUB_CODE))
  case 'interleave':
    for (const [fd, s] of [[1, 'a'], [2, 'b'], [1, 'c'], [2, 'd'], [1, 'e']]) fs.writeSync(fd, s)
    break
  case 'drain': {
    const buf = Buffer.alloc(1 << 16)
    let total = 0
    for (;;) {
      let n
      try {
        n = fs.readSync(0, buf, 0, buf.length, null)
      } catch (e) {
        if (e.code === 'EAGAIN') continue
        if (e.code === 'EOF') break
        throw e
      }
      if (n === 0) break
      total += n
    }
    fs.writeSync(1, String(total))
    break
  }
  default:
    fs.writeSync(1, JSON.stringify(process.argv.slice(2)))
}
`

const SH_STUB = `#!/bin/sh
case "$STUB_MODE" in
  signal) kill -"$STUB_SIGNAL" $$ ;;
  *)      exit 0 ;;
esac
`

let root, launcher, preload

const pkgDir = (name) => path.join(root, 'node_modules', ...name.split('/'))

// Installs <pkg>/bin/<bin> with the given source and mode.
function stub(source, { name = HOST_PKG, bin = HOST_BIN, mode = 0o755 } = {}) {
  const dir = pkgDir(name)
  fs.mkdirSync(path.join(dir, 'bin'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name, version: '0.0.0' }))
  fs.writeFileSync(path.join(dir, 'bin', bin), source)
  fs.chmodSync(path.join(dir, 'bin', bin), mode)
}

before(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'bx-launcher-'))
  launcher = path.join(root, 'bin', 'bx.js')
  fs.mkdirSync(path.dirname(launcher), { recursive: true })
  fs.copyFileSync(SRC, launcher)

  // process.platform and process.arch are configurable, so a --require preload
  // can reach the unsupported-platform and win32 branches on ordinary CI.
  preload = path.join(root, 'fake-platform.js')
  fs.writeFileSync(
    preload,
    "const [platform, arch] = process.env.FAKE_PLATFORM.split(' ')\n" +
      "Object.defineProperty(process, 'platform', { value: platform })\n" +
      "Object.defineProperty(process, 'arch', { value: arch })\n"
  )
})

after(() => fs.rmSync(root, { recursive: true, force: true }))

beforeEach(() => {
  fs.rmSync(path.join(root, 'node_modules'), { recursive: true, force: true })
  stub(NODE_STUB)
})

// Invokes the launcher the way npm's shim does: `node .../bin/bx.js <args>`.
function run(args = [], opts = {}) {
  const r = spawnSync(
    process.execPath,
    [...(opts.fakePlatform ? ['--require', preload] : []), launcher, ...args],
    {
      input: opts.input,
      stdio: opts.stdio ?? [opts.input === undefined ? 'ignore' : 'pipe', 'pipe', 'pipe'],
      maxBuffer: 1 << 26,
      timeout: 30_000,
      env: {
        ...process.env,
        STUB_MODE: 'argv',
        ...(opts.fakePlatform && { FAKE_PLATFORM: opts.fakePlatform }),
        ...opts.env,
      },
    }
  )
  assert.equal(r.signal, null, `the launcher itself was killed by ${r.signal}`)
  return { code: r.status, stdout: String(r.stdout ?? ''), stderr: String(r.stderr ?? '') }
}

const argvOf = (r) => JSON.parse(r.stdout)

// ── argv passthrough ──────────────────────────────────────────────────

test('argv: no args forwards an empty list and prints nothing extra', () => {
  const r = run([])
  assert.equal(r.code, 0)
  assert.equal(r.stdout, '[]')
  assert.equal(r.stderr, '')
})

test('argv: hostile shapes survive byte-for-byte', () => {
  const nasty = [
    '', ' ', 'two words', 'say "hi"', "it's", '--', '-', '-x', '--count=3', '--',
    'café \u{1F600}', 'line1\nline2', 'a\tb', 'trailing\n',
    ';rm -rf /', '&& id', '| wc -l', '$(id)', '`id`', '${HOME}', '*', '~', '\\',
  ]
  const r = run(nasty)
  assert.equal(r.code, 0)
  assert.deepEqual(argvOf(r), nasty)
})

test('argv: no shell is interposed, so substitutions stay literal', () => {
  const r = run(['$(id)', '`id`', '; id'])
  assert.deepEqual(argvOf(r), ['$(id)', '`id`', '; id'])
  assert.doesNotMatch(r.stdout, /uid=/)
})

test('argv: 4000 args round-trip', () => {
  const many = Array.from({ length: 4000 }, (_, i) => `a${i}`)
  assert.deepEqual(argvOf(run(many)), many)
})

// One byte under Linux's MAX_ARG_STRLEN. Anything at or over it is rejected by
// execve before the launcher runs at all, so there is nothing to assert there.
test('argv: a 131071-byte arg passes through', linuxOnly, () => {
  const big = 'x'.repeat(131071)
  assert.deepEqual(argvOf(run([big])), [big])
})

// ── exit status ───────────────────────────────────────────────────────

for (const code of [0, 1, 2, 3, 4, 5, 42, 254, 255]) {
  test(`exit: status ${code} is propagated verbatim`, () => {
    assert.equal(run([], { env: { STUB_MODE: 'exit', STUB_CODE: String(code) } }).code, code)
  })
}

// ── signals ───────────────────────────────────────────────────────────

const SIGNALS = [['SIGHUP', 1], ['SIGINT', 2], ['SIGABRT', 6], ['SIGKILL', 9], ['SIGSEGV', 11], ['SIGTERM', 15]]

for (const [name, signo] of SIGNALS) {
  test(`signal: a child killed by ${name} becomes ${128 + signo}`, posixOnly, () => {
    stub(SH_STUB)
    assert.equal(run([], { env: { STUB_MODE: 'signal', STUB_SIGNAL: String(signo) } }).code, 128 + signo)
  })
}

// Real-time signals have no entry in os.constants.signals, so Node reports
// signal === "" — falsy — with status === null, dropping past the signal branch
// into the fallback. That fallback must not be 1, which is one of bx's own codes.
test("signal: SIGRTMIN is unnameable, so the fallback exits 127 rather than 1", posixOnly, () => {
  stub(SH_STUB)
  assert.equal(run([], { env: { STUB_MODE: 'signal', STUB_SIGNAL: '34' } }).code, 127)
})

// ── stdio ─────────────────────────────────────────────────────────────

test('stdin: piped bytes reach the child', () => {
  assert.equal(run([], { env: { STUB_MODE: 'drain' }, input: Buffer.from('hello') }).stdout, '5')
})

test('stdin: a closed stdin does not hang', () => {
  const r = run([], { env: { STUB_MODE: 'drain' } })
  assert.equal(r.code, 0)
  assert.equal(r.stdout, '0')
})

test('stdin: a 2 MiB payload is not truncated', () => {
  const size = 2 * 1024 * 1024
  const r = run([], { env: { STUB_MODE: 'drain' }, input: Buffer.alloc(size, 0x61) })
  assert.equal(r.stdout, String(size))
})

test('stdio: inherited stdout and stderr stay interleaved and unbuffered', () => {
  const log = path.join(root, 'interleave.log')
  const fd = fs.openSync(log, 'w')
  try {
    run([], { env: { STUB_MODE: 'interleave' }, stdio: ['ignore', fd, fd] })
  } finally {
    fs.closeSync(fd)
  }
  assert.equal(fs.readFileSync(log, 'utf8'), 'abcde')
})

// ── launch failures ───────────────────────────────────────────────────

test('resolve: a missing platform package reports 127 with install guidance', () => {
  fs.rmSync(path.join(root, 'node_modules'), { recursive: true, force: true })
  const r = run(['web', 'x'])
  assert.equal(r.code, 127)
  assert.equal(
    r.stderr.trim(),
    `bx: the optional dependency "${HOST_PKG}" did not install. Try reinstalling. ` +
      'See https://github.com/brave/brave-search-cli#quick-start for install options.'
  )
  assert.equal(r.stdout, '')
})

test('resolve: the package present but its binary missing reports 127', () => {
  fs.rmSync(path.join(pkgDir(HOST_PKG), 'bin', HOST_BIN))
  const r = run([])
  assert.equal(r.code, 127)
  assert.match(r.stderr, /did not install/)
})

test('spawn: a non-executable binary reports 127, not a crash', posixOnly, () => {
  stub(NODE_STUB, { mode: 0o644 })
  const r = run([])
  assert.equal(r.code, 127)
  assert.match(r.stderr, /^bx: failed to run ".*": .*EACCES/m)
})

// ── platform mapping ──────────────────────────────────────────────────

// Deliberately a second copy of the launcher's table: a typo in either is a
// platform that silently resolves nothing.
const SUPPORTED = {
  'darwin arm64': '@brave/brave-search-cli-darwin-arm64',
  'linux x64': '@brave/brave-search-cli-linux-x64',
  'linux arm64': '@brave/brave-search-cli-linux-arm64',
  'win32 x64': '@brave/brave-search-cli-win32-x64',
  'win32 arm64': '@brave/brave-search-cli-win32-arm64',
}

for (const [key, name] of Object.entries(SUPPORTED)) {
  test(`platform: ${key} resolves ${name}`, () => {
    fs.rmSync(path.join(root, 'node_modules'), { recursive: true, force: true })
    const r = run([], { fakePlatform: key })
    assert.equal(r.code, 127)
    assert.ok(r.stderr.includes(`"${name}" did not install`), r.stderr)
  })
}

test('platform: win32 wants bx.exe, so a bare bx does not satisfy it', () => {
  stub(NODE_STUB, { name: SUPPORTED['win32 x64'], bin: 'bx' })
  const r = run([], { fakePlatform: 'win32 x64' })
  assert.equal(r.code, 127)
  assert.match(r.stderr, /did not install/)
})

test('platform: win32 runs bin/bx.exe when present', posixOnly, () => {
  stub(NODE_STUB, { name: SUPPORTED['win32 x64'], bin: 'bx.exe' })
  const r = run(['a', 'b'], { fakePlatform: 'win32 x64' })
  assert.equal(r.code, 0)
  assert.deepEqual(argvOf(r), ['a', 'b'])
})

test('platform: an unknown platform reports 127 with the quick-start link', () => {
  const r = run([], { fakePlatform: 'sunos mips' })
  assert.equal(r.code, 127)
  assert.equal(
    r.stderr.trim(),
    'bx: unsupported platform (sunos mips). ' +
      'See https://github.com/brave/brave-search-cli#quick-start for install options.'
  )
})

test('platform: 32-bit Linux is unsupported', () => {
  const r = run([], { fakePlatform: 'linux ia32' })
  assert.equal(r.code, 127)
  assert.match(r.stderr, /unsupported platform \(linux ia32\)/)
})

// darwin x64 is the one unsupported pair people actually hit: an Intel Mac, or
// Apple Silicon running an x64 Node under Rosetta. Each needs a different fix.
test('platform: darwin x64 distinguishes Rosetta from a real Intel Mac', () => {
  const r = run([], { fakePlatform: 'darwin x64' })
  assert.equal(r.code, 127)
  assert.match(r.stderr, /unsupported platform \(darwin x64\)/)
  assert.match(r.stderr, /Rosetta, reinstall Node as arm64/)
  assert.match(r.stderr, /Intel Mac, build from source/)
})

// ── packaging ─────────────────────────────────────────────────────────

test('source: bin/bx.js parses', () => {
  assert.equal(spawnSync(process.execPath, ['--check', SRC]).status, 0)
})

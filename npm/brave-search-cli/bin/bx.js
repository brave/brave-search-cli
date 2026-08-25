#!/usr/bin/env node
'use strict'

const { spawnSync } = require('node:child_process')
const { signals } = require('node:os').constants

const PLATFORM_PACKAGES = {
  'darwin arm64': '@brave/brave-search-cli-darwin-arm64',
  'darwin x64': '@brave/brave-search-cli-darwin-x64',
  'linux x64': '@brave/brave-search-cli-linux-x64',
  'linux arm64': '@brave/brave-search-cli-linux-arm64',
  'win32 x64': '@brave/brave-search-cli-win32-x64',
  'win32 arm64': '@brave/brave-search-cli-win32-arm64',
}

const HELP = 'See https://github.com/brave/brave-search-cli#quick-start for install options.'

// 127 ("command not found") keeps launch failures out of bx's own 0-5 range.
function fail(message) {
  console.error(`bx: ${message}`)
  process.exit(127)
}

const key = `${process.platform} ${process.arch}`
const pkg = PLATFORM_PACKAGES[key]

if (!pkg) {
  fail(`unsupported platform (${key}). ${HELP}`)
}

const binName = process.platform === 'win32' ? 'bx.exe' : 'bx'

let binPath
try {
  binPath = require.resolve(`${pkg}/bin/${binName}`)
} catch {
  fail(`the optional dependency "${pkg}" did not install. Try reinstalling. ${HELP}`)
}

const result = spawnSync(binPath, process.argv.slice(2), { stdio: 'inherit' })

if (result.error) {
  fail(`failed to run "${binPath}": ${result.error.message}`)
}

// Killed by a signal: the shell's 128 + N convention. Node reports "" for a
// signal it cannot name, so anything indeterminate exits 127, never 0-5.
const signalNumber = signals[result.signal]
process.exit(signalNumber ? 128 + signalNumber : (result.status ?? 127))

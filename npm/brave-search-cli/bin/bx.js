#!/usr/bin/env node
'use strict'

const { spawnSync } = require('node:child_process')

const PLATFORM_PACKAGES = {
  'darwin arm64': '@brave/brave-search-cli-darwin-arm64',
  'linux x64': '@brave/brave-search-cli-linux-x64',
  'linux arm64': '@brave/brave-search-cli-linux-arm64',
  'win32 x64': '@brave/brave-search-cli-win32-x64',
  'win32 arm64': '@brave/brave-search-cli-win32-arm64',
}

function fail(message) {
  console.error(`bx: ${message}`)
  process.exit(1)
}

const key = `${process.platform} ${process.arch}`
const pkg = PLATFORM_PACKAGES[key]

if (!pkg) {
  fail(
    `unsupported platform (${key}). ` +
      'See https://github.com/brave/brave-search-cli#quick-start for other install options.'
  )
}

const binName = process.platform === 'win32' ? 'bx.exe' : 'bx'

let binPath
try {
  binPath = require.resolve(`${pkg}/bin/${binName}`)
} catch {
  fail(
    `the optional dependency "${pkg}" did not install. ` +
      'Try reinstalling, or install directly: https://github.com/brave/brave-search-cli#quick-start'
  )
}

const result = spawnSync(binPath, process.argv.slice(2), { stdio: 'inherit' })

if (result.error) {
  fail(`failed to run "${binPath}": ${result.error.message}`)
}

process.exit(result.status === null ? 1 : result.status)

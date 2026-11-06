BINARY = bx
VERSION = $(shell grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

.PHONY: test
test:
	./tests/run.sh

.PHONY: test-unit
test-unit:
	cargo test

# npm launcher: hermetic, no API key or network needed.
.PHONY: test-npm
test-npm:
	node --test 'npm/brave-search-cli/test/*.test.js'

.PHONY: build
build:
	cargo build --release

# ── Cross-compilation (all via Docker) ────────────────────────────────

.PHONY: linux-amd64
linux-amd64:
	docker build -t $(BINARY)-$@ --build-arg TARGET=x86_64-unknown-linux-musl .
	$(call docker-extract,$(BINARY)-$@,$@)

.PHONY: linux-arm64
linux-arm64:
	docker build -t $(BINARY)-$@ --build-arg TARGET=aarch64-unknown-linux-musl .
	$(call docker-extract,$(BINARY)-$@,$@)

.PHONY: darwin-amd64
darwin-amd64:
	docker build -f Dockerfile.darwin -t $(BINARY)-$@ \
	  --build-arg TARGET=x86_64-apple-darwin .
	$(call docker-extract,$(BINARY)-$@,$@)

.PHONY: darwin-arm64
darwin-arm64:
	docker build -f Dockerfile.darwin -t $(BINARY)-$@ \
	  --build-arg TARGET=aarch64-apple-darwin .
	$(call docker-extract,$(BINARY)-$@,$@)

.PHONY: windows-amd64
windows-amd64:
	docker build -f Dockerfile.windows -t $(BINARY)-$@ \
	  --build-arg TARGET=x86_64-pc-windows-gnu .
	$(call docker-extract,$(BINARY)-$@,$@,.exe)

.PHONY: windows-arm64
windows-arm64:
	docker build -f Dockerfile.windows-arm64 -t $(BINARY)-$@ \
	  --build-arg TARGET=aarch64-pc-windows-gnullvm .
	$(call docker-extract,$(BINARY)-$@,$@,.exe)

.PHONY: dist-all
dist-all: linux-amd64 linux-arm64 darwin-amd64 darwin-arm64 windows-amd64 windows-arm64

# `&&` throughout: any failed step stops before the tag. Runs from a clean main equal to
# origin/main; the atomic push lands both refs or neither, and a refused one is undone locally.
.PHONY: release
release:
ifndef NEW_VERSION
	$(error Usage: make release NEW_VERSION=v0.5.0)
endif
	@printf '%s\n' "$(NEW_VERSION)" | grep -Eqx 'v[0-9]+\.[0-9]+\.[0-9]+' || { echo "NEW_VERSION must be vX.Y.Z (got $(NEW_VERSION))" >&2; exit 1; } && \
	branch=$$(git rev-parse --abbrev-ref HEAD) && \
	[ "$$branch" = "main" ] || { echo "release runs from main, not $$branch" >&2; exit 1; } && \
	[ -z "$$(git status --porcelain)" ] || { echo "working tree is not clean" >&2; exit 1; } && \
	git fetch -q origin main && \
	[ "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main)" ] || { echo "main is not origin/main; pull first" >&2; exit 1; } && \
	version=$$(echo "$(NEW_VERSION)" | sed 's/^v//') && \
	sed -i.bak 's/^version = ".*"/version = "'$$version'"/' Cargo.toml && rm Cargo.toml.bak && \
	cargo check --quiet && \
	git add Cargo.toml Cargo.lock && \
	git commit -m "release $(NEW_VERSION)" && \
	git tag "$(NEW_VERSION)" && \
	{ git push --atomic origin main "$(NEW_VERSION)" || { git tag -d "$(NEW_VERSION)" && git reset -q --keep HEAD~1; echo "push refused; local release commit and tag undone" >&2; exit 1; }; }

.PHONY: clean
clean:
	cargo clean
	rm -rf dist

# One container per target, so `make -j dist-all` jobs cannot clobber each other. The `.sha256`
# is what publish-npm.yml and the install scripts verify.
define docker-extract
	mkdir -p dist
	docker rm -f tmp-$(BINARY)-$(2) 2>/dev/null || true
	docker create --name tmp-$(BINARY)-$(2) $(1) /dev/null
	docker cp tmp-$(BINARY)-$(2):/$(BINARY)$(3) dist/$(BINARY)-$(VERSION)-$(2)$(3)
	docker rm tmp-$(BINARY)-$(2)
	cd dist && sha256sum $(BINARY)-$(VERSION)-$(2)$(3) > $(BINARY)-$(VERSION)-$(2)$(3).sha256
endef

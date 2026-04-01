BINARY = bx
VERSION = $(shell grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

.PHONY: test
test:
	./tests/run.sh

.PHONY: test-unit
test-unit:
	cargo test

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
dist-all: linux-amd64 linux-arm64 darwin-arm64 windows-amd64 windows-arm64

.PHONY: release
release:
ifndef NEW_VERSION
	$(error Usage: make release NEW_VERSION=v0.5.0)
endif
	@version=$$(echo "$(NEW_VERSION)" | sed 's/^v//'); \
	sed -i 's/^version = ".*"/version = "'$$version'"/' Cargo.toml; \
	cargo check --quiet; \
	git add Cargo.toml Cargo.lock; \
	git commit -m "release $(NEW_VERSION)"; \
	git tag "$(NEW_VERSION)"; \
	git push origin HEAD "$(NEW_VERSION)"

.PHONY: clean
clean:
	cargo clean
	rm -rf dist

define docker-extract
	mkdir -p dist
	docker rm -f tmp-$(BINARY) 2>/dev/null || true
	docker create --name tmp-$(BINARY) $(1) /dev/null
	docker cp tmp-$(BINARY):/$(BINARY)$(3) dist/$(BINARY)-$(VERSION)-$(2)$(3)
	docker rm tmp-$(BINARY)
endef

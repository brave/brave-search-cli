ARG TARGET=x86_64-unknown-linux-musl

# Pin by digest to prevent supply-chain tag-swap attacks. Renovate auto-updates.
# Refresh: docker pull rust:1-slim && docker inspect --format='{{index .RepoDigests 0}}' rust:1-slim
FROM rust:1-slim@sha256:cf09adf8c3ebaba10779e5c23ff7fe4df4cccdab8a91f199b0c142c53fef3e1a AS builder
ARG TARGET

RUN apt-get update && apt-get install -y musl-tools gcc-aarch64-linux-gnu gcc-x86-64-linux-gnu && rm -rf /var/lib/apt/lists/*
RUN rustup target add "$TARGET"

ENV CC_aarch64_unknown_linux_musl=aarch64-linux-gnu-gcc
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=aarch64-linux-gnu-gcc
ENV CC_x86_64_unknown_linux_musl=x86_64-linux-gnu-gcc
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=x86_64-linux-gnu-gcc

WORKDIR /src
COPY . .

RUN cargo build --release --target "$TARGET"

FROM scratch
ARG TARGET
COPY --from=builder /src/target/${TARGET}/release/bx /bx
USER 65534:65534
ENTRYPOINT ["/bx"]

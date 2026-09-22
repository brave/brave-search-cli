ARG TARGET=x86_64-unknown-linux-musl

# Pin by digest to prevent supply-chain tag-swap attacks. Renovate auto-updates.
# Refresh: docker pull rust:1-slim && docker inspect --format='{{index .RepoDigests 0}}' rust:1-slim
FROM rust:1-slim@sha256:f47a8de237dcbb0b0ce1099901e60a89728e3d51f24e664b40e947171538ade7 AS builder
ARG TARGET

# The MSRV, pinned in every release Dockerfile so a digest bump cannot split compilers.
RUN rustup toolchain install 1.98.1 --profile minimal && rustup default 1.98.1
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

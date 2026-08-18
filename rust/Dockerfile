# Multi-stage build: compile in a full Rust image, run in a minimal one.
# The final image doesn't need the Rust toolchain at all, just the
# compiled binaries, so it stays small.
#
# Pinned to rust:1.85. If a dependency update pushes the minimum
# supported Rust version higher and this stops compiling, bump this tag
# to a newer one, or run `rustup update` locally and check what you're
# actually building with.

FROM rust:1-slim-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build every binary in src/bin/ in one pass.
RUN cargo build --release

FROM debian:bookworm-slim

# ca-certificates is required for outbound HTTPS calls (Anthropic,
# OpenAI, the MCP server, etc.), the slim base image doesn't include it.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/direct_rpc /app/target/release/chainkit_fetch /app/target/release/chainkit_mcp_agent /app/target/release/advisor /app/target/release/rag /app/

CMD ["sh", "-c", "echo 'Specify a binary to run, e.g.: docker compose run rust ./rag \"your question\"'"]

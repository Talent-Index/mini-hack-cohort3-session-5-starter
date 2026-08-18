# Session 5 Starter — JavaScript

Retrieval-augmented generation, plus Docker support. Builds on
Session 4's Smart Wallet Advisor and the model-provider pattern from
Sessions 1 and 2.

## Setup

**With Docker** (no local Node install needed):

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY, GLACIER_API_KEY, WALLET_ADDRESS at minimum
cd .. && docker compose up -d chroma
docker compose run --rm javascript npm run rag -- "What is the C-Chain?"
```

**Without Docker:**

```bash
npm install
cp .env.example .env
# fill in ANTHROPIC_API_KEY, GLACIER_API_KEY, WALLET_ADDRESS at minimum
# start Chroma yourself: docker run -p 8000:8000 chromadb/chroma
```

## Files

| File | What it does |
|---|---|
| `model-provider.js` | Same provider abstraction from Session 2, carried forward unchanged |
| `direct-rpc.js` | Method 1: raw RPC via `ethers.js`, `getBalance`/`getBlock`/`getTransactionCount` |
| `chainkit-fetch.js` | Method 2: structured wallet history via the real `@avalanche-sdk/chainkit` SDK |
| `chainkit-mcp-agent.js` | ChainKit running as an MCP server, wired into a tool-calling agent |
| `advisor.js` | Session 4: the Smart Wallet Advisor, fetch, normalize, summarize, with a human-in-the-loop checkpoint and audit logging |
| `rag.js` | Session 5: retrieval-augmented generation, chunk, embed, store, retrieve, generate, grounded answers with citations |
| `normalize.js` | Shared wei-to-AVAX, hex-to-decimal, Unix-to-ISO8601 conversion, used by all data methods |

## Running each one

```bash
npm run direct-rpc          # Method 1, no API key needed beyond the RPC endpoint itself
npm run fetch-transactions  # Method 2, needs GLACIER_API_KEY
npm run mcp-agent           # ChainKit as MCP, needs the mcp-server running separately first
npm run advisor -- <wallet-address>       # Session 4: the full Smart Wallet Advisor
npm run rag -- "your question"            # Session 5: the RAG agent, needs Chroma running
```

For `mcp-agent`, start the ChainKit MCP server in another terminal
first (its CLI route is `start`, over the SSE transport):

```bash
npx -y @avalanche-sdk/chainkit start --transport sse --port 3000
# or, equivalently:
npm run chainkit-mcp-server
```

It serves the SSE transport at `http://localhost:3000/sse`. Put that URL
in `CHAINKIT_MCP_URL` in your `.env` before running the agent.

For `rag`, Chroma needs to be running first, either via
`docker compose up -d chroma` from the repo root, or locally with
`docker run -p 8000:8000 chromadb/chroma`.

## A note on embeddings

The official `chromadb` npm client computes embeddings locally, in this
process, using a small bundled model (`all-MiniLM-L6-v2`). The first
time you run `rag.js`, Chroma downloads that model, a few tens of
megabytes, so the very first call is slower and needs internet access.
After that it's cached and runs offline. `rag.js` never calls an
embeddings API directly, this all happens inside the `chromadb` package.

## Model provider

Same as Session 2: `MODEL_PROVIDER` in `.env` picks the provider
(`anthropic`, `openai`, `gemini`, or `ollama`), defaulting to
`anthropic` if unset. The `anthropic` and `openai` paths both implement
tool calling, required for `chainkit-mcp-agent.js` to work — the
model-provider layer normalizes tool calls to one shape, so the agent
runs the same on either. `gemini` and `ollama` are plain text chat for
now (they accept a `tools` argument but ignore it). This doesn't affect
`rag.js`, which doesn't use tools — it just calls whichever provider you
have active for a normal grounded answer.

## Submission

1. Test everything yourself, confirm your RAG agent answers correctly from your documents and refuses anything outside them.
2. Screenshot the working test, including at least one grounded answer with a citation, and one correct refusal.
3. Open your PR, screenshot that too.
4. Post on X with both screenshots, tag **@code_mwangi** and **@AvaxAfrica**.
5. Copy your post link, submit it on the quest page once it's live.

Post in the Week 3 WhatsApp group for anything you get stuck on.

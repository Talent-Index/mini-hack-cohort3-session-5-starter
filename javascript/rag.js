// Retrieval-Augmented Generation (RAG), end to end.
//
// Five steps: chunk your documents (skipped here, our five docs are
// already short enough to index whole), embed each one, store the
// embeddings in Chroma, retrieve the closest matches at query time, and
// generate an answer grounded in those matches, with citations.
//
// A note on embeddings: the official Chroma client computes embeddings
// locally, in this process, using a small bundled model (all-MiniLM-L6-v2).
// The first time you run this, Chroma downloads that model, a few tens
// of megabytes, so the very first call is slower and needs internet
// access. After that it's cached and runs offline.

import { ChromaClient } from "chromadb";
import { createModelClient } from "./model-provider.js";

// Chroma connection, configurable so this same code works whether
// Chroma is running locally (CHROMA_HOST defaults to localhost) or as
// a sibling service in Docker Compose (CHROMA_HOST=chroma there).
const chromaHost = process.env.CHROMA_HOST || "localhost";
const chromaPort = process.env.CHROMA_PORT || "8000";
const chromaClient = new ChromaClient({
  host: chromaHost,
  port: Number(chromaPort),
  ssl: false,
});

// Five documents, five sources, matching what the curriculum asks for.
// In your own project, swap these for your actual docs, README, API
// reference, whatever your agent should be able to answer questions
// about.
const documents = [
  {
    id: "avax-cchain",
    text: "The C-Chain is Avalanche's EVM-compatible smart contract chain. This is where Solidity contracts, USDC, and most DeFi activity actually run.",
  },
  {
    id: "chainkit-readme",
    text: "ChainKit is Avalanche's official on-chain data SDK. It provides structured, paginated access to wallet transaction history, and can also run as its own MCP server.",
  },
  {
    id: "daraja-api",
    text: "The Daraja API is Safaricom's integration platform for M-Pesa. It exposes endpoints for STK push, C2B, B2C, and transaction status queries.",
  },
  {
    id: "ethers-core",
    text: "ethers.js is a library for interacting with Ethereum-compatible chains. It handles wallets, providers, contract calls, and transaction signing.",
  },
  {
    id: "project-readme",
    text: "Mini Hack Cohort 3 starter repo: a provider-agnostic agent starter supporting Claude, OpenAI, Gemini, and Ollama through one shared interface.",
  },
];

// Step 1 (chunk, already done above) + Step 2 and 3 (embed and store).
// getOrCreateCollection makes this safe to run more than once, it won't
// duplicate the collection on a second run.
async function buildKnowledgeBase() {
  const collection = await chromaClient.getOrCreateCollection({
    name: "mini-hack-docs",
  });

  await collection.add({
    ids: documents.map((d) => d.id),
    documents: documents.map((d) => d.text),
    // no "embeddings" field: we're letting Chroma's default embedding
    // function handle step 2 for us, this is the whole point of using
    // a vector database instead of writing embedding code by hand.
  });

  return collection;
}

// Step 4: retrieve. Embeds the question the same way, using the same
// default embedding function, then finds the closest chunks by cosine
// similarity under the hood.
async function queryKnowledgeBase(collection, question) {
  const results = await collection.query({
    queryTexts: [question],
    nResults: 3,
  });

  // Chroma returns parallel arrays (one per query, we only sent one),
  // zip them back into a friendlier shape: one object per chunk.
  return results.documents[0].map((text, i) => ({
    text,
    id: results.ids[0][i],
    distance: results.distances[0][i], // lower distance = closer match
  }));
}

// Step 5, part one: build a prompt that forces the model to answer only
// from what we retrieved, and to cite which chunk it used. This single
// instruction is what turns "the model made something up" into "the
// model told me exactly where this came from."
function buildGroundedPrompt(question, chunks) {
  const context = chunks
    .map((c, i) => `[${i + 1}] (source: ${c.id})\n${c.text}`)
    .join("\n\n");

  return `Answer the question using only the context below. Cite which source number supports each claim. If the context does not cover the question, say you don't know.

Context:
${context}

Question: ${question}`;
}

// Step 5, part two: call the model and print the grounded answer.
async function main() {
  const collection = await buildKnowledgeBase();
  const client = await createModelClient("openai");

  const question = process.argv[2] || "What is the C-Chain?";
  const chunks = await queryKnowledgeBase(collection, question);

  const response = await client.generateText({
    systemPrompt:
      "You are a documentation assistant. Only answer from the provided context.",
    messages: [
      { role: "user", content: buildGroundedPrompt(question, chunks) },
    ],
  });

  console.log(response.text);
}

main().catch((err) => {
  console.error("RAG agent error:", err.message);
  // Set an error exit code and let Node shut down on its own instead of
  // calling process.exit(1). A hard exit here races the native embedding
  // library (onnxruntime, loaded by Chroma's default embedder) as it tears
  // down, which on macOS surfaces as "mutex lock failed: Invalid argument"
  // and aborts the process. Letting the event loop drain lets that library
  // unload cleanly, so you get the error message above without the crash.
  process.exitCode = 1;
});

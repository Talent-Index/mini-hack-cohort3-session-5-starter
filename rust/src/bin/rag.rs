//! Retrieval-Augmented Generation (RAG), end to end.
//!
//! Five steps: chunk your documents (skipped here, our five docs are
//! already short enough to index whole), embed each one, store the
//! embeddings in Chroma, retrieve the closest matches at query time,
//! and generate an answer grounded in those matches, with citations.
//!
//! A note on embeddings, Rust specific: there is no official Chroma
//! Rust crate, and going through Chroma's REST API directly means
//! embeddings must be computed before they are sent, the server does
//! not embed raw text for you over plain REST the way the JS and
//! Python SDKs do locally. This file computes embeddings by calling
//! OpenAI's embeddings endpoint, which means you need OPENAI_API_KEY
//! set in your `.env` regardless of which MODEL_PROVIDER you have chat
//! generation pointed at, embeddings and chat completion are separate
//! capabilities.
//!
//! The Chroma REST endpoints below (tenants, databases, collections,
//! add, query) were verified against a real running Chroma v2 server
//! during development, not guessed from documentation.

use mini_hack_session5::modelprovider::{new_model_client, Message};
use serde_json::{json, Value};
use std::env;

const CHROMA_TENANT: &str = "default_tenant";
const CHROMA_DATABASE: &str = "default_database";
const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";
const EMBEDDING_MODEL: &str = "text-embedding-3-small";

struct Doc {
    id: &'static str,
    text: &'static str,
}

// Five documents, five sources, matching what the curriculum asks for.
// In your own project, swap these for your actual docs, README, API
// reference, whatever your agent should be able to answer questions
// about.
const DOCUMENTS: [Doc; 5] = [
    Doc { id: "avax-cchain", text: "The C-Chain is Avalanche's EVM-compatible smart contract chain. This is where Solidity contracts, USDC, and most DeFi activity actually run." },
    Doc { id: "chainkit-readme", text: "ChainKit is Avalanche's official on-chain data SDK. It provides structured, paginated access to wallet transaction history, and can also run as its own MCP server." },
    Doc { id: "daraja-api", text: "The Daraja API is Safaricom's integration platform for M-Pesa. It exposes endpoints for STK push, C2B, B2C, and transaction status queries." },
    Doc { id: "ethers-core", text: "ethers.js is a library for interacting with Ethereum-compatible chains. It handles wallets, providers, contract calls, and transaction signing." },
    Doc { id: "project-readme", text: "Mini Hack Cohort 3 starter repo: a provider-agnostic agent starter supporting Claude, OpenAI, Gemini, and Ollama through one shared interface." },
];

struct Chunk {
    id: String,
    text: String,
}

// Step 2: embed. Calls OpenAI's embeddings endpoint for one or more
// strings at once, batching is both faster and cheaper than one call
// per document.
async fn embed_texts(client: &reqwest::Client, texts: &[&str]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY is not set, embeddings need it even if your chat model is a different provider")?;

    let body = json!({ "model": EMBEDDING_MODEL, "input": texts });

    let resp = client
        .post(OPENAI_EMBEDDINGS_URL)
        .bearer_auth(&api_key)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(format!("OpenAI embeddings request failed with status {}", resp.status()).into());
    }

    let data: Value = resp.json().await?;
    let embeddings = data["data"]
        .as_array()
        .ok_or("unexpected embeddings response shape")?
        .iter()
        .map(|item| {
            item["embedding"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect()
        })
        .collect();

    Ok(embeddings)
}

// Chroma connection, configurable so this same code works whether
// Chroma is running locally (CHROMA_HOST defaults to localhost) or as
// a sibling service in Docker Compose (CHROMA_HOST=chroma there).
fn chroma_base_url() -> String {
    let host = env::var("CHROMA_HOST").unwrap_or_else(|_| "localhost".to_string());
    let port = env::var("CHROMA_PORT").unwrap_or_else(|_| "8000".to_string());
    format!("http://{host}:{port}/api/v2")
}

async fn chroma_post(client: &reqwest::Client, path: &str, body: &Value) -> Result<Value, Box<dyn std::error::Error>> {
    let url = format!("{}{path}", chroma_base_url());
    let resp = client.post(&url).json(body).send().await?;

    if !resp.status().is_success() {
        return Err(format!("Chroma request to {path} failed with status {}", resp.status()).into());
    }

    // some Chroma endpoints (like /add) return no body on success
    let text = resp.text().await?;
    if text.is_empty() {
        return Ok(json!({}));
    }
    Ok(serde_json::from_str(&text)?)
}

// Step 3: store. get_or_create semantics, if the collection already
// exists this just returns its id instead of erroring.
async fn get_or_create_collection(client: &reqwest::Client, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections");
    let result = chroma_post(client, &path, &json!({ "name": name, "get_or_create": true })).await?;
    result["id"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Chroma did not return a collection id".into())
}

async fn add_documents(
    client: &reqwest::Client,
    collection_id: &str,
    ids: &[&str],
    texts: &[&str],
    embeddings: &[Vec<f64>],
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections/{collection_id}/add");
    chroma_post(client, &path, &json!({ "ids": ids, "documents": texts, "embeddings": embeddings })).await?;
    Ok(())
}

// Step 4: retrieve. Embeds the question the same way, then asks Chroma
// for the closest matches by cosine similarity.
async fn query_knowledge_base(
    client: &reqwest::Client,
    collection_id: &str,
    question: &str,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let question_embeddings = embed_texts(client, &[question]).await?;

    let path = format!("/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections/{collection_id}/query");
    let result = chroma_post(client, &path, &json!({ "query_embeddings": question_embeddings, "n_results": 3 })).await?;

    // Chroma returns parallel arrays nested one level for "one array per
    // query", we only sent one query, so we read index 0 throughout.
    let ids = result["ids"][0].as_array().ok_or("no ids in response")?;
    let docs = result["documents"][0].as_array().ok_or("no documents in response")?;

    let chunks = ids
        .iter()
        .zip(docs.iter())
        .map(|(id, text)| Chunk {
            id: id.as_str().unwrap_or_default().to_string(),
            text: text.as_str().unwrap_or_default().to_string(),
        })
        .collect();

    Ok(chunks)
}

// Step 5, part one: build a prompt that forces the model to answer only
// from what we retrieved, and to cite which chunk it used. This single
// instruction is what turns "the model made something up" into "the
// model told me exactly where this came from."
fn build_grounded_prompt(question: &str, chunks: &[Chunk]) -> String {
    let context = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| format!("[{}] (source: {})\n{}", i + 1, c.id, c.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        "Answer the question using only the context below. Cite which source number supports each claim. If the context does not cover the question, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let question = env::args().nth(1).unwrap_or_else(|| "What is the C-Chain?".to_string());

    let http = reqwest::Client::new();

    // Steps 2 and 3: embed and store all five documents.
    let ids: Vec<&str> = DOCUMENTS.iter().map(|d| d.id).collect();
    let texts: Vec<&str> = DOCUMENTS.iter().map(|d| d.text).collect();

    let doc_embeddings = embed_texts(&http, &texts).await?;
    let collection_id = get_or_create_collection(&http, "mini-hack-docs").await?;
    add_documents(&http, &collection_id, &ids, &texts, &doc_embeddings).await?;

    // Step 4: retrieve.
    let chunks = query_knowledge_base(&http, &collection_id, &question).await?;

    // Step 5: build the grounded prompt and call the model.
    let client = new_model_client(None).map_err(|e| e.to_string())?;
    let messages = vec![Message {
        role: "user".to_string(),
        content: build_grounded_prompt(&question, &chunks),
    }];
    let response = client
        .generate_text("You are a documentation assistant. Only answer from the provided context.", &messages, &[])
        .await?;

    println!("{}", response.text);

    Ok(())
}

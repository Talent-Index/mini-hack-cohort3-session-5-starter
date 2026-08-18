// The Smart Wallet Advisor, end to end.
//
// Five steps: fetch a wallet's transaction history via ChainKit (same
// approach as chainkit-fetch.js), normalize it, build a summarization
// prompt, send it to the model, and return a structured plain-English
// summary. Includes a human-in-the-loop checkpoint before anything gets
// flagged for review, and an audit log entry for every decision the
// agent makes along the way.

import "dotenv/config";
import { Avalanche } from "@avalanche-sdk/chainkit";
import { normalizeMany } from "./normalize.js";
import { createModelClient } from "./model-provider.js";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const avalancheSDK = new Avalanche({ apiKey: process.env.GLACIER_API_KEY });
const rl = readline.createInterface({ input, output });

// Step 1: fetch. Same ChainKit call as chainkit-fetch.js, just pulling
// more history (50 transactions instead of 10) since a meaningful
// summary needs more than a handful of data points to work with.
async function getWalletHistory(walletAddress) {
  const  transactions  = await avalancheSDK.data.evm.address.transactions.list({
    chainId: "43113", // Fuji testnet
    address: walletAddress,
    pageSize: 50,
  });
  // Step 2: normalize. Same normalize.js from Session 3, unchanged,
  // this is the whole point of having built it once, reusable everywhere.
  return normalizeMany(transactions.result.transactions);
}

// Step 3: the summarization prompt. Specific about exactly what we want
// back, total AVAX in and out, most frequent counterparty, primary
// token, one notable pattern, not just "summarize this wallet." A vague
// prompt here gets a vague summary, same lesson from Session 1.
function buildSummaryPrompt(normalizedTxs) {
  return `Summarize this wallet's activity. Include:
- total AVAX in and out
- most frequent counterparty
- primary token used
- one notable pattern

Wallet transactions (normalized JSON):
${JSON.stringify(normalizedTxs, null, 2)}`;
}

// Human-in-the-loop checkpoint. Shows the agent's reasoning and the
// action it wants to take, then waits for an explicit human decision
// before anything happens. This is the one pattern the whole session is
// built around, the agent reasons, the human decides, the code executes.
async function confirmAction(reasoning, action) {
  console.log(`\nAgent reasoning: ${reasoning}`);
  console.log(`Proposed action: ${action}`);
  const answer = await rl.question("Approve? (y/n): ");
  return answer.trim().toLowerCase() === "y";
}

// Every decision the agent makes, the fetch, the flag, the human's
// answer, gets logged as one structured entry: what happened, why, and
// when. This is what lets you debug a bad decision after the fact and
// prove what actually happened, not just what you assume happened.
function logDecision({ reasoning, tool, params, result }) {
  const entry = {
    timestamp: new Date().toISOString(),
    reasoning,
    tool,
    params,
    result,
  };
  console.log(JSON.stringify(entry));
  // production: append to a durable log store, not just stdout
}

async function main() {
  const walletAddress = process.argv[2];
  if (!walletAddress) throw new Error("Usage: node advisor.js <wallet-address>");

  // Steps 1 and 2: fetch and normalize.
  const normalizedTxs = await getWalletHistory(walletAddress);
  logDecision({
    reasoning: "Fetched wallet history",
    tool: "chainkit.listTransactions",
    params: { walletAddress },
    result: `${normalizedTxs.length} transactions`,
  });

  // Steps 3, 4, and 5: build the prompt, call the model, print the summary.
  const client = await createModelClient();
  const response = await client.generateText({
    systemPrompt: "You are a wallet analyst. Be precise and concise.",
    messages: [{ role: "user", content: buildSummaryPrompt(normalizedTxs) }],
  });
  console.log(response.text);

  // Human-in-the-loop: anything 5 AVAX or larger gets flagged and needs
  // explicit approval before we log it as reviewed. Nothing here
  // executes a transaction, this is intentionally the simplest possible
  // HITL pattern, show reasoning, wait for a yes or no.
  const largeTx = normalizedTxs.find((t) => t.amount >= 5);
  if (largeTx) {
    const approved = await confirmAction(
      `This wallet sent ${largeTx.amount} AVAX to ${largeTx.to}, unusually large for this pattern.`,
      "Flag transaction for manual review"
    );
    logDecision({
      reasoning: "Large transaction detected",
      tool: "confirmAction",
      params: { hash: largeTx.hash },
      result: approved ? "approved" : "rejected",
    });
  }

  rl.close();
}

main().catch((err) => {
  console.error("Advisor error:", err.message);
  process.exit(1);
});

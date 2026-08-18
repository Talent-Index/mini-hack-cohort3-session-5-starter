// Method 2: The ChainKit SDK
//
// Avalanche's official on-chain data SDK. Structured, paginated results,
// no manual RPC parsing. Needs a free Glacier API key from avacloud.io.

import "dotenv/config";
import { Avalanche } from "@avalanche-sdk/chainkit";
import { normalizeMany } from "./normalize.js";

async function main() {
  const walletAddress = process.env.WALLET_ADDRESS;
  const apiKey = process.env.GLACIER_API_KEY;
  if (!walletAddress) throw new Error("Set WALLET_ADDRESS in your .env first.");
  if (!apiKey) throw new Error("Set GLACIER_API_KEY in your .env first. Get a free one at avacloud.io.");

  const avalancheSDK = new Avalanche({ apiKey });

  const  transactions  = await avalancheSDK.data.evm.address.transactions.list({
    chainId: "43113", // Fuji testnet
    address: walletAddress,
    pageSize: 10,
  });

  console.log(`Found ${transactions.result.transactions.length} transactions\n`);

  const normalized = normalizeMany(transactions.result.transactions);
  for (const tx of normalized) {
    console.log(`${tx.status === "success" ? "OK" : "FAILED"}  ${tx.amount} ${tx.token}  ${tx.timestamp}  ${tx.hash}`);
  }
}

main().catch((err) => {
  console.error("ChainKit fetch error:", err.message);
  process.exit(1);
});

// ChainKit as an MCP server, wired into an agent
//
// Before running this, start the ChainKit MCP server in another terminal:
//   npx -y @avalanche-sdk/chainkit start --transport sse --port 3000
// (or: npm run chainkit-mcp-server)
// It serves the SSE transport at http://localhost:3000/sse — put that URL
// in CHAINKIT_MCP_URL in your .env.
//
// This agent then asks a plain-English question, the model decides to
// call a ChainKit tool, and the tool call is forwarded straight to the
// running MCP server, no manual SDK calls in this file at all. Works with
// any tool-capable provider (anthropic or openai) — the model-provider
// layer normalizes tool calls to one shape.

import "dotenv/config";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { createModelClient } from "./model-provider.js";

const SYSTEM_PROMPT = "You are Mini Hack Assistant. Use tools when they genuinely help; otherwise answer directly.";

async function connectChainkitMcp() {
  const url = process.env.CHAINKIT_MCP_URL;
  if (!url) throw new Error("Set CHAINKIT_MCP_URL in your .env first, e.g. http://localhost:3000/sse from the running mcp-server.");

  const mcpClient = new Client({ name: "mini-hack-agent", version: "1.0.0" });
  const transport = new SSEClientTransport(new URL(url));
  await mcpClient.connect(transport);
  return mcpClient;
}

async function main() {
  const rl = readline.createInterface({ input, output });
  const messages = [];

  const client = await createModelClient();
  const mcpClient = await connectChainkitMcp();
  const { tools } = await mcpClient.listTools();

  console.log(`Mini Hack on-chain agent using ${client.provider}, connected to ChainKit MCP. Type 'exit' to quit.\n`);

  while (true) {
    const userInput = await rl.question("You: ");
    if (userInput.trim().toLowerCase() === "exit") break;

    messages.push({ role: "user", content: userInput });

    let response = await client.generateText({ systemPrompt: SYSTEM_PROMPT, messages, tools });

    while (response.toolCalls.length > 0) {
      messages.push({ role: "assistant", content: response.text, toolCalls: response.toolCalls });

      const toolResults = [];
      for (const call of response.toolCalls) {
        try {
          const result = await mcpClient.callTool({ name: call.name, arguments: call.input });
          toolResults.push({ id: call.id, name: call.name, content: JSON.stringify(result.content) });
        } catch (err) {
          toolResults.push({ id: call.id, name: call.name, content: `Error: ${err.message}`, isError: true });
        }
      }

      messages.push({ role: "tool", toolResults });
      response = await client.generateText({ systemPrompt: SYSTEM_PROMPT, messages, tools });
    }

    console.log(`\nAssistant: ${response.text}\n`);
    messages.push({ role: "assistant", content: response.text, toolCalls: response.toolCalls });
  }

  rl.close();
}

main().catch((err) => {
  console.error("Agent error:", err.message);
  process.exit(1);
});

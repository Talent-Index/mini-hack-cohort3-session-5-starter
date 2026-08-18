// model-provider.js
//
// One factory, one interface, four providers. createModelClient() gives you
// back { provider, generateText }. Every provider implements the same
// shape so chat.js — and anything you build on top of it — never needs to
// know which model is actually running underneath.
//
// Provider selection: pass a name explicitly — createModelClient("openai")
// — or leave it blank and it reads MODEL_PROVIDER from .env, falling back
// to "anthropic" if that's not set either.
//
// generateText({ systemPrompt, messages, tools }) always returns:
//   { text, toolCalls, stopReason, raw }
//
//   text       — the assistant's reply text ("" if it only called tools)
//   toolCalls  — [{ id, name, input }], normalized regardless of provider.
//                Empty array if the model didn't call a tool, OR if this
//                provider doesn't support tool calling yet (see below).
//   stopReason — the provider's own reason string, kept as-is, not normalized
//   raw        — the full untouched response, in case you need provider-specific detail
//
// MESSAGE FORMAT — provider-neutral, translated per provider
// Callers speak ONE shape, and each client translates it to its own wire
// format. A message is one of:
//   { role: "user",      content: "text" }
//   { role: "assistant", content: "text", toolCalls: [{ id, name, input }] }
//   { role: "tool",      toolResults: [{ id, name, content, isError }] }
// `tools` is the raw MCP tool list ([{ name, description, inputSchema }]);
// each client converts it to the provider's tool schema, so the agent never
// hand-writes Anthropic's input_schema or OpenAI's function wrappers.
//
// TOOL-CALLING SUPPORT
// The Anthropic and OpenAI clients both implement tools end to end: they
// accept `tools`, translate the schema, forward tool calls, and normalize
// the response back to toolCalls: [{ id, name, input }]. Gemini and Ollama
// accept a `tools` argument without erroring but ignore it and always
// return toolCalls: [] — their function-calling shapes aren't wired up yet.
// stopReason is normalized to "tool_use" whenever the model asked for a
// tool, so an agent loop can branch on it regardless of provider.

const SUPPORTED_PROVIDERS = ["anthropic", "openai", "gemini", "ollama"];

// ---------------------------------------------------------------------------
// Message + tool translators: provider-neutral shape -> provider wire format
// ---------------------------------------------------------------------------

function safeJsonParse(str) {
  if (!str) return {};
  try {
    return JSON.parse(str);
  } catch {
    return {};
  }
}

function toAnthropicTools(tools) {
  if (!tools?.length) return undefined;
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema ??
      tool.input_schema ?? { type: "object", properties: {} },
  }));
}

function toAnthropicMessages(messages) {
  return messages.map((message) => {
    if (message.role === "tool") {
      return {
        role: "user",
        content: message.toolResults.map((result) => ({
          type: "tool_result",
          tool_use_id: result.id,
          content: result.content,
          ...(result.isError ? { is_error: true } : {}),
        })),
      };
    }

    if (message.role === "assistant") {
      const blocks = [];
      if (message.content) blocks.push({ type: "text", text: message.content });
      for (const call of message.toolCalls ?? []) {
        blocks.push({
          type: "tool_use",
          id: call.id,
          name: call.name,
          input: call.input ?? {},
        });
      }
      return { role: "assistant", content: blocks.length ? blocks : message.content ?? "" };
    }

    return { role: "user", content: message.content };
  });
}

function toOpenAITools(tools) {
  if (!tools?.length) return undefined;
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema ??
        tool.input_schema ?? { type: "object", properties: {} },
    },
  }));
}

function toOpenAIMessages(messages) {
  const out = [];
  for (const message of messages) {
    if (message.role === "tool") {
      for (const result of message.toolResults) {
        out.push({ role: "tool", tool_call_id: result.id, content: result.content });
      }
      continue;
    }

    if (message.role === "assistant") {
      const entry = { role: "assistant", content: message.content ?? "" };
      if (message.toolCalls?.length) {
        entry.content = message.content || null;
        entry.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: JSON.stringify(call.input ?? {}) },
        }));
      }
      out.push(entry);
      continue;
    }

    out.push({ role: "user", content: message.content });
  }
  return out;
}

function getConfiguredProvider() {
  const provider =
    process.env.MODEL_PROVIDER?.trim().toLowerCase() || "anthropic";

  if (!SUPPORTED_PROVIDERS.includes(provider)) {
    throw new Error(
      `Unsupported MODEL_PROVIDER "${provider}". Use one of: ${SUPPORTED_PROVIDERS.join(", ")}`,
    );
  }

  return provider;
}

// Generic text extraction for providers that don't (yet) return structured
// tool calls — tries the common shapes a chat-completion response takes.
function extractText(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value.map(extractText).filter(Boolean).join("\n");
  }

  if (!value || typeof value !== "object") return "";

  if (typeof value.text === "string") return value.text;
  if (typeof value.content === "string") return value.content;
  if (Array.isArray(value.content))
    return value.content.map(extractText).filter(Boolean).join("\n");
  if (typeof value.message?.content === "string") return value.message.content;
  if (Array.isArray(value.message?.content)) {
    return value.message.content.map(extractText).filter(Boolean).join("\n");
  }

  return "";
}

function normalizeResponse(response) {
  if (typeof response === "string") return response;

  const candidates = [
    response?.content,
    response?.choices?.[0]?.message?.content,
    response?.message?.content,
    response?.text,
    response?.result,
    response?.reply,
    response?.response,
  ];

  for (const candidate of candidates) {
    const text = extractText(candidate);
    if (text) return text;
  }

  throw new Error("Unable to extract text from model response.");
}

async function createAnthropicClient() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY is not set.");
  }

  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic({ apiKey });

  return {
    provider: "anthropic",
    async generateText({ systemPrompt, messages, tools }) {
      const response = await client.messages.create({
        model: process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6",
        max_tokens: Number(process.env.MAX_TOKENS || 1024),
        system: systemPrompt,
        tools: toAnthropicTools(tools),
        messages: toAnthropicMessages(messages),
      });

      // Precise extraction, not the generic guesser below — we know this
      // shape exactly, and tool_use blocks need to survive the round trip.
      const textBlock = response.content.find((b) => b.type === "text");
      const toolCalls = response.content
        .filter((b) => b.type === "tool_use")
        .map((b) => ({ id: b.id, name: b.name, input: b.input }));

      return {
        text: textBlock ? textBlock.text : "",
        toolCalls,
        stopReason: response.stop_reason,
        raw: response,
      };
    },
  };
}

async function createOpenAIClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set.");
  }

  const { default: OpenAI } = await import("openai");
  const client = new OpenAI({ apiKey });

  return {
    provider: "openai",
    async generateText({ systemPrompt, messages, tools }) {
      const response = await client.chat.completions.create({
        model: process.env.OPENAI_MODEL || "gpt-4.1",
        max_tokens: Number(process.env.MAX_TOKENS || 1024),
        messages: [
          { role: "system", content: systemPrompt },
          ...toOpenAIMessages(messages),
        ],
        tools: toOpenAITools(tools),
      });

      const choice = response.choices?.[0];
      const message = choice?.message;

      // Normalize OpenAI's tool_calls (function name + JSON-string arguments)
      // into the same { id, name, input } shape the Anthropic client returns.
      const toolCalls = (message?.tool_calls ?? [])
        .filter((call) => call.type === "function")
        .map((call) => ({
          id: call.id,
          name: call.function.name,
          input: safeJsonParse(call.function.arguments),
        }));

      return {
        text: message?.content ?? "",
        toolCalls,
        // Normalize to "tool_use" so agents can branch the same way they do
        // for Anthropic; OpenAI's own reason for this is "tool_calls".
        stopReason: toolCalls.length ? "tool_use" : choice?.finish_reason ?? "unknown",
        raw: response,
      };
    },
  };
}

async function createGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set.");
  }

  const { GoogleGenAI } = await import("@google/genai");
  const client = new GoogleGenAI({ apiKey });

  return {
    provider: "gemini",
    async generateText({ systemPrompt, messages }) {
      // Tool calling not yet implemented for this provider — see the note
      // at the top of this file. Plain text chat only, for now.
      const response = await client.models.generateContent({
        model: process.env.GEMINI_MODEL || "gemini-2.5-flash",
        config: { systemInstruction: systemPrompt },
        contents: messages.map((message) => ({
          role: message.role === "assistant" ? "model" : "user",
          parts: [{ text: message.content }],
        })),
      });

      return {
        text: normalizeResponse(response),
        toolCalls: [],
        stopReason: response.candidates?.[0]?.finishReason ?? "unknown",
        raw: response,
      };
    },
  };
}

async function createOllamaClient() {
  const baseUrl = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
  const model = process.env.OLLAMA_MODEL || "llama3.1";

  return {
    provider: "ollama",
    async generateText({ systemPrompt, messages }) {
      // Tool calling not yet implemented for this provider — see the note
      // at the top of this file. Plain text chat only, for now.
      const response = await fetch(`${baseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          stream: false,
          messages: [{ role: "system", content: systemPrompt }, ...messages],
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Ollama request failed with status ${response.status} — is "ollama serve" running, and have you run "ollama pull ${model}"?`,
        );
      }

      const data = await response.json();
      return {
        text: normalizeResponse(data),
        toolCalls: [],
        stopReason: data.done_reason ?? "unknown",
        raw: data,
      };
    },
  };
}

export async function createModelClient(providerOverride) {
  const provider =
    providerOverride?.trim().toLowerCase() || getConfiguredProvider();

  switch (provider) {
    case "anthropic":
      return createAnthropicClient();
    case "openai":
      return createOpenAIClient();
    case "gemini":
      return createGeminiClient();
    case "ollama":
      return createOllamaClient();
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

export { SUPPORTED_PROVIDERS, normalizeResponse, extractText };

import {anthropic} from "@ai-sdk/anthropic";
import {deepinfra} from "@ai-sdk/deepinfra";
import {createGoogleGenerativeAI} from "@ai-sdk/google";
import {openai} from "@ai-sdk/openai";
import {ModelRequest, TypedModelRequest} from "@korabench/core";
import {toJsonSchema} from "@valibot/to-json-schema";
import {
  gateway,
  generateObject,
  generateText,
  jsonSchema,
  LanguageModel,
} from "ai";
import * as v from "valibot";
import {createLogRetryHandler, RetryOptions, withRetry} from "../retry.js";
import {createFallbackModel} from "./fallbackModel.js";
import {Model} from "./model.js";
import {ModelConfig, resolveModelConfig} from "./modelConfig.js";

export interface ModelOptions {
  retry?: RetryOptions;
}

// ---------------------------------------------------------------------------
// Direct providers.
//
// A models.json entry can be served directly by its provider's AI SDK package
// instead of the Vercel AI Gateway. The provider is detected from the
// `<provider>/` segment of the entry's `model` field (e.g. "openai/gpt-4o",
// "deepinfra/Qwen/Qwen3-32B"), and the direct route is taken only when the
// provider's API key is present in the environment:
//
//     openai     → OPENAI_API_KEY
//     anthropic  → ANTHROPIC_API_KEY
//     google     → GEMINI_API_KEY
//     deepinfra  → DEEPINFRA_API_KEY (OpenAI-compatible completions API)
//
// Otherwise the same slug falls back to the gateway — which keys you export
// decides the routing, per provider. Both routes share the request/retry
// plumbing below; only the underlying LanguageModel differs.
// ---------------------------------------------------------------------------

interface DirectProvider {
  envVar: string;
  factory: (modelId: string) => LanguageModel;
}

const DIRECT_PROVIDERS: Record<string, DirectProvider> = {
  openai: {envVar: "OPENAI_API_KEY", factory: openai},
  anthropic: {envVar: "ANTHROPIC_API_KEY", factory: anthropic},
  google: {
    envVar: "GEMINI_API_KEY",
    // @ai-sdk/google's default instance reads GOOGLE_GENERATIVE_AI_API_KEY;
    // pass the key explicitly so GEMINI_API_KEY is the single source.
    factory: id =>
      createGoogleGenerativeAI({apiKey: process.env.GEMINI_API_KEY})(id),
  },
  deepinfra: {envVar: "DEEPINFRA_API_KEY", factory: deepinfra},
};

/**
 * Return the provider-package LanguageModel for a config whose provider is a
 * known direct provider with its API key set, or undefined (the caller then
 * uses the gateway). Model ids may themselves contain slashes (e.g.
 * deepinfra's "Qwen/Qwen3-32B"), so split on the first slash only.
 */
export function resolveDirectModel(
  config: ModelConfig
): LanguageModel | undefined {
  const slash = config.model.indexOf("/");
  if (slash === -1) {
    return undefined;
  }
  const provider = DIRECT_PROVIDERS[config.model.slice(0, slash)];
  if (!provider || !process.env[provider.envVar]) {
    return undefined;
  }
  return provider.factory(config.model.slice(slash + 1));
}

const defaultRetryOptions: RetryOptions = {
  maxRetries: 5,
  initialDelayMs: 1000,
  maxDelayMs: 60000,
  backoffMultiplier: 2,
  jitterFactor: 0.2,
};

function buildRetryOptions(
  label: string,
  options?: ModelOptions
): Required<Pick<RetryOptions, "onRetry">> & RetryOptions {
  return {
    ...defaultRetryOptions,
    ...options?.retry,
    onRetry: options?.retry?.onRetry ?? createLogRetryHandler(label),
  };
}

// The Vercel AI Gateway corrupts structured-output responses for Anthropic
// (tool arguments dropped, returned as "{}") and for Google (thinking tags
// leak into text when structuredOutputs is off). Bypass generateObject for
// both providers and extract JSON from the plain-text response. Direct
// provider calls are unaffected and use native structured output.
function extractJson(text: string): string {
  const withoutThink = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  const fenceMatch = withoutThink.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch?.[1]) return fenceMatch[1].trim();
  const start = withoutThink.indexOf("{");
  const end = withoutThink.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) return withoutThink;
  return withoutThink.slice(start, end + 1);
}

export function createGatewayModelChain(
  modelsJsonPath: string,
  modelSlugs: readonly string[],
  options?: ModelOptions
): Model {
  if (modelSlugs.length === 0) {
    throw new Error("createGatewayModelChain: at least one slug required.");
  }
  return createFallbackModel(
    modelSlugs.map(slug => ({
      label: slug,
      model: createGatewayModel(modelsJsonPath, slug, options),
    }))
  );
}

export function createGatewayModel(
  modelsJsonPath: string,
  modelSlug: string,
  options?: ModelOptions
): Model {
  const config = resolveModelConfig(modelsJsonPath, modelSlug);

  // Env-driven routing: direct provider package when its key is set, gateway
  // otherwise. Same slug, same plumbing — only the LanguageModel differs.
  const directModel = resolveDirectModel(config);
  const model: LanguageModel = directModel ?? gateway(config.model);

  const retryOptions = buildRetryOptions(config.model, options);
  const providerOptions = config.providerOptions as
    | Record<string, Record<string, never>>
    | undefined;

  return {
    async getTextResponse(request: ModelRequest): Promise<string> {
      const maxTokens = request.maxTokens ?? config.maxTokens;
      const temperature = request.temperature ?? config.temperature;

      const result = await withRetry(
        () =>
          generateText({
            model,
            system: request.messages.find(m => m.role === "system")?.content,
            messages: request.messages
              .filter(m => m.role !== "system")
              .map(m => ({
                role: m.role as "user" | "assistant",
                content: m.content,
              })),
            maxOutputTokens: maxTokens,
            temperature,
            providerOptions,
            maxRetries: 0,
          }),
        retryOptions
      );

      return result.text;
    },

    async getStructuredResponse<T>(request: TypedModelRequest<T>): Promise<T> {
      const outputSchema = toJsonSchema(request.outputType);
      const maxTokens = request.maxTokens ?? config.maxTokens;
      const temperature = request.temperature ?? config.temperature;
      const systemMessage = request.messages.find(
        m => m.role === "system"
      )?.content;
      const userMessages = request.messages
        .filter(m => m.role !== "system")
        .map(m => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        }));

      // Gateway-only workaround (see extractJson above); direct provider
      // calls use native structured output below.
      if (
        !directModel &&
        (config.model.startsWith("google/") ||
          config.model.startsWith("anthropic/"))
      ) {
        const schemaInstruction =
          "Respond with a single JSON object that strictly conforms to this JSON Schema. " +
          "Output JSON only — no prose, no code fences, no <think> tags.\n\n" +
          JSON.stringify(outputSchema);
        const combinedSystem = systemMessage
          ? `${systemMessage}\n\n${schemaInstruction}`
          : schemaInstruction;

        return withRetry(async () => {
          const result = await generateText({
            model,
            system: combinedSystem,
            messages: userMessages,
            maxOutputTokens: maxTokens,
            temperature,
            providerOptions,
            maxRetries: 0,
          });

          const parsed = JSON.parse(extractJson(result.text));
          return v.parse(request.outputType, parsed);
        }, retryOptions);
      }

      return withRetry(async () => {
        const result = await generateObject({
          model,
          system: systemMessage,
          messages: userMessages,
          schema: jsonSchema(outputSchema),
          maxOutputTokens: maxTokens,
          temperature,
          providerOptions,
          maxRetries: 0,
        });

        return v.parse(request.outputType, result.object);
      }, retryOptions);
    },
  };
}

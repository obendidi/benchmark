import {afterEach, describe, expect, it, vi} from "vitest";
import {createGatewayModel, resolveDirectModel} from "../gatewayModel.js";

describe("resolveDirectModel", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns undefined when the provider's key is not set", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "");
    expect(
      resolveDirectModel({model: "openrouter/openai/gpt-4o"})
    ).toBeUndefined();
  });

  it("returns a model when the provider's key is set", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "test-key");
    expect(
      resolveDirectModel({model: "openrouter/openai/gpt-4o"})
    ).toBeDefined();
  });

  it("splits openrouter model ids on the first slash only", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "test-key");
    // OpenRouter ids keep their author prefix, so the id itself contains a slash.
    expect(
      resolveDirectModel({model: "openrouter/deepseek/deepseek-v3.2"})
    ).toBeDefined();
  });

  it("routes cerebras models off CEREBRAS_API_KEY", () => {
    vi.stubEnv("CEREBRAS_API_KEY", "test-key");
    expect(resolveDirectModel({model: "cerebras/gemma-4-31b"})).toBeDefined();
    vi.stubEnv("CEREBRAS_API_KEY", "");
    expect(resolveDirectModel({model: "cerebras/gemma-4-31b"})).toBeUndefined();
  });

  it("returns undefined for every other provider and for malformed ids", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "test-key");
    vi.stubEnv("CEREBRAS_API_KEY", "test-key");
    // Only openrouter and cerebras route directly: these are gateway slugs.
    for (const model of [
      "openai/gpt-4o",
      "anthropic/claude-sonnet-4.6",
      "google/gemini-2.5-flash",
      "deepinfra/Qwen/Qwen3-32B",
      "vertex-anthropic/claude-haiku-4-5",
      "no-provider-segment",
    ]) {
      expect(resolveDirectModel({model})).toBeUndefined();
    }
  });
});

describe("createGatewayModel with raw direct-provider slugs", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("accepts a raw <provider>/<model> slug without a models.json entry", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "test-key");
    // The bogus registry path proves models.json is never read on this path.
    expect(
      createGatewayModel(
        "/nonexistent/models.json",
        "openrouter/openai/gpt-oss-120b"
      )
    ).toBeDefined();
  });

  it("fails fast when the raw slug's provider key is missing", () => {
    vi.stubEnv("OPENROUTER_API_KEY", "");
    expect(() =>
      createGatewayModel(
        "/nonexistent/models.json",
        "openrouter/openai/gpt-oss-120b"
      )
    ).toThrow(/OPENROUTER_API_KEY/);
  });
});

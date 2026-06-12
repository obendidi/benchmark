import {afterEach, describe, expect, it, vi} from "vitest";
import {createGatewayModel, resolveDirectModel} from "../gatewayModel.js";

describe("resolveDirectModel", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns undefined when the provider's key is not set", () => {
    vi.stubEnv("OPENAI_API_KEY", "");
    expect(resolveDirectModel({model: "openai/gpt-4o"})).toBeUndefined();
  });

  it("returns a model when the provider's key is set", () => {
    vi.stubEnv("OPENAI_API_KEY", "test-key");
    expect(resolveDirectModel({model: "openai/gpt-4o"})).toBeDefined();
  });

  it("returns undefined for unknown providers and malformed ids", () => {
    vi.stubEnv("OPENAI_API_KEY", "test-key");
    expect(
      resolveDirectModel({model: "deepseek/deepseek-v3.2"})
    ).toBeUndefined();
    expect(resolveDirectModel({model: "no-provider-segment"})).toBeUndefined();
  });

  it("routes each provider segment off its own env var", () => {
    vi.stubEnv("ANTHROPIC_API_KEY", "test-key");
    vi.stubEnv("GEMINI_API_KEY", "");
    expect(
      resolveDirectModel({model: "anthropic/claude-sonnet-4-6"})
    ).toBeDefined();
    expect(
      resolveDirectModel({model: "google/gemini-2.5-flash"})
    ).toBeUndefined();
  });

  it("splits deepinfra model ids on the first slash only", () => {
    vi.stubEnv("DEEPINFRA_API_KEY", "test-key");
    expect(
      resolveDirectModel({model: "deepinfra/Qwen/Qwen3-32B"})
    ).toBeDefined();
  });

  it("routes cerebras models off CEREBRAS_API_KEY", () => {
    vi.stubEnv("CEREBRAS_API_KEY", "test-key");
    expect(resolveDirectModel({model: "cerebras/gpt-oss-120b"})).toBeDefined();
    vi.stubEnv("CEREBRAS_API_KEY", "");
    expect(
      resolveDirectModel({model: "cerebras/gpt-oss-120b"})
    ).toBeUndefined();
  });
});

describe("createGatewayModel with raw direct-provider slugs", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("accepts a raw <provider>/<model> slug without a models.json entry", () => {
    vi.stubEnv("CEREBRAS_API_KEY", "test-key");
    // The bogus registry path proves models.json is never read on this path.
    expect(
      createGatewayModel("/nonexistent/models.json", "cerebras/gpt-oss-120b")
    ).toBeDefined();
  });

  it("fails fast when the raw slug's provider key is missing", () => {
    vi.stubEnv("CEREBRAS_API_KEY", "");
    expect(() =>
      createGatewayModel("/nonexistent/models.json", "cerebras/gpt-oss-120b")
    ).toThrow(/CEREBRAS_API_KEY/);
  });
});

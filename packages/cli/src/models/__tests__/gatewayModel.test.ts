import {afterEach, describe, expect, it, vi} from "vitest";
import {resolveDirectModel} from "../gatewayModel.js";

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
    vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "");
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
});

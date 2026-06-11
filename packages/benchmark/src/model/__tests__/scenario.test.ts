import * as v from "valibot";
import {describe, expect, it} from "vitest";
import {createScenario, createScenarioSeed} from "../../__tests__/fixtures.js";
import {Scenario} from "../scenario.js";
import {ScenarioSeed} from "../scenarioSeed.js";

describe("Scenario.io language", () => {
  it("parses scenarios without a language (pre-existing corpora)", () => {
    const parsed = v.parse(Scenario.io, createScenario());

    expect(parsed.language).toBeUndefined();
  });

  it("parses scenarios with a supported language", () => {
    const parsed = v.parse(Scenario.io, createScenario({language: "fr"}));

    expect(parsed.language).toBe("fr");
  });

  it("rejects unsupported languages", () => {
    expect(() =>
      v.parse(Scenario.io, {...createScenario(), language: "de"})
    ).toThrow();
  });
});

describe("ScenarioSeed.io 4to6 age band", () => {
  it("parses seeds in the 4to6 band with childAge down to 4", () => {
    const seed = createScenarioSeed({ageRange: "4to6", childAge: 4});

    const parsed = v.parse(ScenarioSeed.io, seed);

    expect(parsed.ageRange).toBe("4to6");
    expect(parsed.childAge).toBe(4);
  });

  it("rejects childAge below 4", () => {
    expect(() =>
      v.parse(ScenarioSeed.io, createScenarioSeed({childAge: 3}))
    ).toThrow();
  });
});

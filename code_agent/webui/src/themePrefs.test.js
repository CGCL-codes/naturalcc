import assert from "node:assert/strict";
import test from "node:test";

import {
  defaultTheme,
  loadThemePref,
  nextTheme,
  normalizeTheme,
  saveThemePref
} from "./themePrefs.js";

function makeStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    }
  };
}

test("normalizeTheme accepts only light and dark", () => {
  assert.equal(normalizeTheme("light"), "light");
  assert.equal(normalizeTheme("dark"), "dark");
  assert.equal(normalizeTheme("system"), defaultTheme);
  assert.equal(normalizeTheme(""), defaultTheme);
});

test("loadThemePref falls back to dark for missing or invalid storage values", () => {
  assert.equal(loadThemePref(makeStorage()), defaultTheme);
  assert.equal(loadThemePref(makeStorage({ "code-agent.theme": "light" })), "light");
  assert.equal(loadThemePref(makeStorage({ "code-agent.theme": "invalid" })), defaultTheme);
});

test("saveThemePref persists normalized theme and nextTheme toggles two modes", () => {
  const storage = makeStorage();

  assert.equal(saveThemePref(storage, "light"), "light");
  assert.equal(loadThemePref(storage), "light");
  assert.equal(saveThemePref(storage, "invalid"), defaultTheme);
  assert.equal(nextTheme("dark"), "light");
  assert.equal(nextTheme("light"), "dark");
  assert.equal(nextTheme("invalid"), "light");
});

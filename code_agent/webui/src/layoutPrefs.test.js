import assert from "node:assert/strict";
import test from "node:test";

import {
  clampLayoutValue,
  defaultLayoutPrefs,
  loadLayoutPrefs,
  saveLayoutPrefs,
  toLayoutStyle
} from "./layoutPrefs.js";

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

test("clampLayoutValue keeps panel sizes inside explicit bounds", () => {
  assert.equal(clampLayoutValue(120, { min: 220, max: 520, fallback: 260 }), 220);
  assert.equal(clampLayoutValue(900, { min: 220, max: 520, fallback: 260 }), 520);
  assert.equal(clampLayoutValue(360, { min: 220, max: 520, fallback: 260 }), 360);
  assert.equal(clampLayoutValue(Number.NaN, { min: 220, max: 520, fallback: 260 }), 260);
});

test("loadLayoutPrefs falls back for missing or invalid persisted values", () => {
  const storage = makeStorage({
    "code-agent.layout": JSON.stringify({
      sidebarWidth: 999,
      agentStackHeight: "bad",
      drawerWidth: 300
    })
  });

  assert.deepEqual(loadLayoutPrefs(storage), {
    ...defaultLayoutPrefs,
    sidebarWidth: 520,
    drawerWidth: 300
  });
});

test("saveLayoutPrefs stores clamped values and toLayoutStyle emits css variables", () => {
  const storage = makeStorage();
  const saved = saveLayoutPrefs(storage, {
    sidebarWidth: 180,
    agentStackHeight: 420,
    drawerWidth: 700
  });

  assert.deepEqual(saved, {
    sidebarWidth: 220,
    agentStackHeight: 420,
    drawerWidth: 620
  });
  assert.deepEqual(loadLayoutPrefs(storage), saved);
  assert.deepEqual(toLayoutStyle(saved), {
    "--sidebar-width": "220px",
    "--agent-stack-height": "420px",
    "--drawer-width": "620px"
  });
});

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { test } from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const electron = process.env.ELECTRON_BIN;

test("compiled Electron host starts the native CPU sidecar", { skip: !electron }, async () => {
  const { ELECTRON_RUN_AS_NODE: _ignored, ...environment } = process.env;
  const child = spawn(electron, ["--no-sandbox", "--headless", "--disable-gpu", path.join(root, "dist", "main.js"), "--load-model"], {
    cwd: root,
    env: { ...environment, ELECTRON_DISABLE_SANDBOX: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => { stdout += data; });
  child.stderr.on("data", (data) => { stderr += data; });
  const exitCode = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code) => resolve(code));
  });
  assert.equal(exitCode, 0, stderr);
  const messages = stdout.trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(messages[0].sidecar.type, "pong");
  assert.equal(messages[1].loaded.type, "loaded");
  assert.equal(messages[1].loaded.backend, "native-cpu");
});

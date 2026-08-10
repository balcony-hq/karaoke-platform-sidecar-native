import { app } from "electron";

import { NativeClient, resolveNativeCommands } from "./native-client.js";

async function main(): Promise<void> {
  await app.whenReady();
  let client: NativeClient | undefined;
  let lastError: unknown;
  try {
    let pong;
    for (const command of resolveNativeCommands()) {
      const candidate = new NativeClient(command);
      try {
        pong = await candidate.request({ type: "ping" });
        client = candidate;
        break;
      } catch (error) {
        lastError = error;
        await candidate.close();
      }
    }
    if (!client || !pong) {
      throw new Error(`native GPU and CPU sidecar startup failed: ${String(lastError ?? "unknown error")}`);
    }
    process.stdout.write(`${JSON.stringify({ ok: true, sidecar: pong })}\n`);
    if (process.argv.includes("--load-model")) {
      const loaded = await client.request({ type: "load" });
      process.stdout.write(`${JSON.stringify({ ok: true, loaded })}\n`);
    }
    await client.close();
    app.quit();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    await client?.close();
    app.quit();
    process.exitCode = 1;
  }
}

void main();

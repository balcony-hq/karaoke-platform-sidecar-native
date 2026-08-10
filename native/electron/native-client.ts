import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";

export type NativeCommand = { command: string; args: string[]; cwd: string; provider: "gpu" | "cpu" };
type Message = Record<string, unknown> & { id?: number };

function firstExisting(paths: string[]): string | undefined {
  return paths.find((candidate) => existsSync(candidate));
}

export function resolveNativeCommands(): NativeCommand[] {
  const root = process.env.VOCALARC_NATIVE_ROOT ?? path.resolve(import.meta.dirname, "..");
  const suffix = process.platform === "win32" ? ".exe" : "";
  const candidates: NativeCommand[] = [];
  for (const provider of ["gpu", "cpu"] as const) {
    const name = provider === "gpu" ? `vocalarc-separation-gpu${suffix}` : `vocalarc-separation${suffix}`;
    const binary = firstExisting([
      provider === "gpu" ? process.env.VOCALARC_NATIVE_SIDECAR ?? "" : "",
      path.join(root, "build", name),
    ]);
    if (binary) candidates.push({ command: binary, args: [], cwd: root, provider });
  }
  if (candidates.length === 0) throw new Error(`native CPU sidecar is missing below ${root}/build`);
  return candidates;
}

export class NativeClient {
  readonly child: ChildProcessWithoutNullStreams;
  private nextId = 1;
  private readonly pending = new Map<number, { resolve: (message: Message) => void; reject: (error: Error) => void }>();
  private readonly lines;

  constructor(command: NativeCommand) {
    this.child = spawn(command.command, command.args, {
      cwd: command.cwd,
      env: { ...process.env, LC_ALL: "C" },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    this.lines = createInterface({ input: this.child.stdout });
    this.lines.on("line", (line: string) => this.onLine(line));
    this.child.stderr.on("data", (data: Buffer) => process.stderr.write(`[native-cpu] ${data.toString()}`));
    this.child.on("error", (error) => this.failAll(error));
    this.child.on("exit", (code, signal) => {
      if (code !== 0) this.failAll(new Error(`native sidecar exited with code=${code} signal=${signal}`));
    });
  }

  private onLine(line: string): void {
    let message: Message;
    try { message = JSON.parse(line) as Message; }
    catch { this.failAll(new Error(`native sidecar emitted invalid JSON: ${line}`)); return; }
    if (typeof message.id !== "number") return;
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.ok === false) pending.reject(new Error(String(message.error ?? "native sidecar failed")));
    else pending.resolve(message);
  }

  private failAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  request(request: Record<string, unknown>): Promise<Message> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(`${JSON.stringify({ id, ...request })}\n`);
    });
  }

  async close(): Promise<void> {
    if (!this.child.killed) {
      try { await this.request({ type: "shutdown" }); } catch { /* process may already have exited */ }
      this.child.kill();
    }
    this.lines.close();
  }
}

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execa, ExecaChildProcess } from 'execa';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
export const projectRoot = path.resolve(scriptDir, '..');

export function resolvePath(...parts: string[]): string {
  return path.resolve(...parts);
}

export function nowStamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(
    d.getHours(),
  )}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

export function ensureSingleCudaVisible(): void {
  const current = process.env.CUDA_VISIBLE_DEVICES;
  if (current && current.includes(',')) {
    throw new Error('CUDA_VISIBLE_DEVICES must expose exactly one GPU (no comma).');
  }
  if (!current) {
    process.env.CUDA_VISIBLE_DEVICES = '0';
  }
}

export function vllmServeArgs({
  model,
  port,
  gpuMemoryUtilization,
  maxModelLen = 2048,
  dtype = 'float16',
  tensorParallelSize = 1,
}: {
  model: string;
  port: string;
  gpuMemoryUtilization: number | string;
  maxModelLen?: number;
  dtype?: string;
  tensorParallelSize?: number;
}): string[] {
  return [
    'serve',
    model,
    '--dtype',
    dtype,
    '--tensor-parallel-size',
    String(tensorParallelSize),
    '--max-model-len',
    String(maxModelLen),
    '--gpu-memory-utilization',
    String(gpuMemoryUtilization),
    '--disable-log-requests',
    '--port',
    String(port),
  ];
}

export function spawnLogged(
  command: string,
  args: string[],
  options: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    logFile?: string;
    name?: string;
  } = {},
): ExecaChildProcess {
  const { cwd, env, logFile, name = command } = options;
  const child = execa(command, args, {
    cwd,
    env,
    stdio: 'pipe',
    detached: true,
    reject: false,
  });

  if (logFile) {
    const stream = fs.createWriteStream(logFile, { flags: 'a' });
    if (child.stdout) {
      child.stdout.pipe(process.stdout);
      child.stdout.pipe(stream);
    }
    if (child.stderr) {
      child.stderr.pipe(process.stderr);
      child.stderr.pipe(stream);
    }
  } else {
    if (child.stdout) child.stdout.pipe(process.stdout);
    if (child.stderr) child.stderr.pipe(process.stderr);
  }

  child.on('error', (err) => {
    console.error(`[${name}] spawn error:`, err.message);
  });

  return child;
}

export async function waitForHttpOk(
  url: string,
  {
    timeoutMs = 120_000,
    intervalMs = 1000,
  }: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastError: Error | null = null;

  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { method: 'GET' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (err) {
      lastError = err as Error;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }

  const reason = lastError ? lastError.message : 'timeout';
  throw new Error(`Timeout waiting for ${url} (${reason})`);
}

export async function waitForPorts(
  baseUrls: string[],
  options?: { timeoutMs?: number; intervalMs?: number },
): Promise<void> {
  for (const baseUrl of baseUrls) {
    await waitForHttpOk(`${baseUrl}/v1/models`, options);
  }
}

export function killProcessTree(child?: ExecaChildProcess, timeoutMs = 5000): void {
  if (!child?.pid) return;
  try {
    process.kill(-child.pid, 'SIGTERM');
  } catch (err) {
    const error = err as NodeJS.ErrnoException;
    if (error.code !== 'ESRCH') {
      console.error('Failed to terminate process group:', error.message);
    }
    return;
  }

  const started = Date.now();
  const interval = setInterval(() => {
    if (Date.now() - started > timeoutMs) {
      clearInterval(interval);
      try {
        process.kill(-child.pid, 'SIGKILL');
      } catch (err) {
        const error = err as NodeJS.ErrnoException;
        if (error.code !== 'ESRCH') {
          console.error('Failed to force kill process group:', error.message);
        }
      }
    }
  }, 200);
}

export function parseBenchOutput(text: string): Record<string, number> {
  const patterns: Record<string, RegExp> = {
    totalElapsed: /Total elapsed wall time \(s\):\s*([0-9.]+)/,
    totalTokens: /Total generated tokens:\s*(\d+)/,
    tokensPerSecond: /Tokens per second:\s*([0-9.]+)/,
    p50Ms: /P50 latency \(ms\):\s*([0-9.]+)/,
    p99Ms: /P99 latency \(ms\):\s*([0-9.]+)/,
  };

  const result: Record<string, number> = {};
  for (const [key, regex] of Object.entries(patterns)) {
    const match = text.match(regex);
    if (match) result[key] = Number(match[1]);
  }
  return result;
}

export function ensureExists(label: string, filePath: string): void {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
}

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import https from 'node:https';
import {execa} from 'execa';
import type { Subprocess } from 'execa';
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
export const projectRoot = path.resolve(scriptDir, '..');

export function resolvePath(...parts: string[]): string {
  return path.resolve(...parts);
}

export function applyLocalHfCacheEnv(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  if (env.HF_HOME || env.HF_HUB_CACHE || env.TRANSFORMERS_CACHE) {
    return env;
  }
  return {
    ...env,
    HF_HOME: path.resolve(projectRoot, '.cache', 'huggingface'),
  };
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

export function resolveVllmCommand(): { command: string; argsPrefix: string[] } {
  const vllmPython = process.env.VLLM_PYTHON;
  if (vllmPython && vllmPython.trim()) {
    return { command: vllmPython, argsPrefix: ['-m', 'vllm.entrypoints.cli.main'] };
  }

  const vllmBin = process.env.VLLM_BIN;
  if (vllmBin && vllmBin.trim()) {
    return { command: vllmBin, argsPrefix: [] };
  }

  return { command: 'vllm', argsPrefix: [] };
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
): Subprocess {
  const { cwd, env, logFile, name = command } = options;
  const child = execa(command, args, {
    cwd,
    env,
    stdio: 'pipe',
    detached: true,
    reject: false,
  });

  if (logFile) {
    // Use a large highWaterMark so vLLM/child is not blocked when logging a lot at startup
    const stream = fs.createWriteStream(logFile, {
      flags: 'a',
      highWaterMark: 512 * 1024,
    });
    if (child.stdout) {
      child.stdout.pipe(stream);
    }
    if (child.stderr) {
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

async function getHttpStatus(url: string, timeoutMs: number): Promise<number> {
  return await new Promise((resolve, reject) => {
    const target = new URL(url);
    const lib = target.protocol === 'https:' ? https : http;
    const req = lib.request(
      {
        method: 'GET',
        hostname: target.hostname,
        port: target.port,
        path: `${target.pathname}${target.search}`,
        timeout: timeoutMs,
      },
      (res) => {
        res.resume();
        resolve(res.statusCode ?? 0);
      },
    );

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy(new Error('timeout'));
    });
    req.end();
  });
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

  const start = Date.now();
  while (Date.now() < deadline) {
    try {
      const status = await getHttpStatus(url, Math.min(5000, timeoutMs));
      if (status >= 200 && status < 300) return;
      lastError = new Error(`HTTP ${status}`);
    } catch (err) {
      lastError = err as Error;
    }
    const elapsed = Math.round((Date.now() - start) / 1000);
    process.stderr.write(`  waiting for ${url} ... ${elapsed}s\n`);
    await new Promise((r) => setTimeout(r, intervalMs));
  }

  const reason = lastError ? lastError.message : 'timeout';
  throw new Error(`Timeout waiting for ${url} (${reason})`);
}

export async function waitForHttpReady(
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
      const status = await getHttpStatus(url, Math.min(5000, timeoutMs));
      if (status > 0) return;
      lastError = new Error(`HTTP ${status}`);
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

export function killProcessTree(child?: Subprocess, timeoutMs = 5000): void {
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
        if (!child.pid) return;
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

type GpuSample = {
  timestampMs: number;
  utilGpu: number;
  utilMem: number;
  memUsedMiB: number;
  memTotalMiB: number;
  powerW: number;
  tempC: number;
};

type GpuStats = {
  samples: number;
  avg: Partial<GpuSample>;
  max: Partial<GpuSample>;
};

function parseGpuSample(line: string): GpuSample | null {
  const parts = line.split(',').map((v) => v.trim());
  if (parts.length < 6) return null;
  const [utilGpu, utilMem, memUsedMiB, memTotalMiB, powerW, tempC] = parts.map(
    (v) => Number(v),
  );
  if (![utilGpu, utilMem, memUsedMiB, memTotalMiB, powerW, tempC].every(Number.isFinite)) {
    return null;
  }
  return {
    timestampMs: Date.now(),
    utilGpu,
    utilMem,
    memUsedMiB,
    memTotalMiB,
    powerW,
    tempC,
  };
}

export function startGpuSampler(intervalMs = 1000): {
  stop: () => Promise<GpuStats | null>;
} {
  let stopped = false;
  let inFlight = false;
  const samples: GpuSample[] = [];

  const timer = setInterval(async () => {
    if (stopped || inFlight) return;
    inFlight = true;
    try {
      const result = await execa('nvidia-smi', [
        '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits',
      ]);
      const line = (result.stdout || '').split('\n')[0]?.trim();
      if (line) {
        const sample = parseGpuSample(line);
        if (sample) samples.push(sample);
      }
    } catch {
      // Ignore sampling errors; keep benchmark running.
    } finally {
      inFlight = false;
    }
  }, intervalMs);

  return {
    stop: async () => {
      stopped = true;
      clearInterval(timer);
      if (!samples.length) return null;

      const avg: Partial<GpuSample> = {};
      const max: Partial<GpuSample> = {};
      const keys: Array<keyof GpuSample> = [
        'utilGpu',
        'utilMem',
        'memUsedMiB',
        'memTotalMiB',
        'powerW',
        'tempC',
      ];

      for (const key of keys) {
        const values = samples.map((s) => s[key]);
        const sum = values.reduce((a, b) => a + b, 0);
        avg[key] = sum / values.length;
        max[key] = Math.max(...values);
      }

      return { samples: samples.length, avg, max };
    },
  };
}

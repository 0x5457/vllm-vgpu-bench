#!/usr/bin/env node
import fs from 'node:fs';
import { execa } from 'execa';
import { Command } from 'commander';
import {
  DEFAULT_BASE_HOST,
  DEFAULT_GPU_UTIL_DOUBLE,
  DEFAULT_GPU_UTIL_SINGLE,
  DEFAULT_HYPERVISOR_PORT,
  DEFAULT_MODEL,
  DEFAULT_SHM_PATH,
} from './config.js';
import {
  ensureSingleCudaVisible,
  ensureExists,
  killProcessTree,
  nowStamp,
  parseBenchOutput,
  projectRoot,
  resolvePath,
  spawnLogged,
  vllmServeArgs,
  waitForPorts,
} from './lib.js';

const program = new Command();

program
  .name('run_matrix')
  .description('Run vLLM limiter matrix benchmarks (baseline + limiter).')
  .option('--modes <list>', 'Modes to test (single,double)', 'single,double')
  .option('--utils <list>', 'Limiter target-util list', '0.3,0.5,0.7')
  .option('--skip-baseline', 'Skip no-limiter runs')
  .option('--results-dir <path>', 'Output directory base', 'results')
  .option('--python <path>', 'Python executable', 'python')
  .option('--concurrency <n>', 'Bench concurrency', '1')
  .option('--total-requests <n>', 'Total requests per run', '60')
  .option('--max-tokens <n>', 'Max tokens per request', '128')
  .option('--timeout-s <n>', 'Request timeout seconds', '120')
  .option('--hypervisor-bin <path>', 'Hypervisor binary path override')
  .option('--limiter-so <path>', 'libcuda_limiter.so path override')
  .parse(process.argv);

const options = program.opts();

const modes = String(options.modes)
  .split(',')
  .map((v: string) => v.trim())
  .filter(Boolean);

const utils = String(options.utils)
  .split(',')
  .map((v: string) => Number(v.trim()))
  .filter((v: number) => Number.isFinite(v));

const skipBaseline = Boolean(options.skipBaseline);
const resultsBase = String(options.resultsDir);
const pythonBin = String(options.python);
const concurrency = Number(options.concurrency);
const totalRequests = Number(options.totalRequests);
const maxTokens = Number(options.maxTokens);
const timeoutSeconds = Number(options.timeoutS);

const resultsRoot = resolvePath(projectRoot, resultsBase, nowStamp());
fs.mkdirSync(resultsRoot, { recursive: true });

const vgpuRoot =
  process.env.VGPU_ROOT ?? resolvePath(projectRoot, '..', 'vgpu.rs');
const hypervisorBin =
  options.hypervisorBin ??
  process.env.HYPERVISOR_BIN ??
  resolvePath(vgpuRoot, 'target', 'release', 'hypervisor');
const limiterSo =
  options.limiterSo ??
  process.env.CUDA_LIMITER_SO ??
  resolvePath(vgpuRoot, 'target', 'release', 'libcuda_limiter.so');

const summaryRows: Array<Record<string, unknown>> = [];
const childProcesses: Array<ReturnType<typeof spawnLogged>> = [];

function registerChild(child: ReturnType<typeof spawnLogged> | null) {
  if (!child) return;
  childProcesses.push(child);
}

async function cleanup() {
  while (childProcesses.length) {
    const child = childProcesses.pop();
    killProcessTree(child);
  }
}

process.on('SIGINT', async () => {
  await cleanup();
  process.exit(130);
});

process.on('SIGTERM', async () => {
  await cleanup();
  process.exit(143);
});

function buildBenchArgs(baseUrls: string[]): string[] {
  return [
    'bench_client.py',
    '--base-urls',
    baseUrls.join(','),
    '--total-requests',
    String(totalRequests),
    '--concurrency',
    String(concurrency),
    '--max-tokens',
    String(maxTokens),
    '--timeout-s',
    String(timeoutSeconds),
  ];
}

async function runBench(runDir: string, baseUrls: string[]) {
  const logFile = resolvePath(runDir, 'bench.log');
  const args = buildBenchArgs(baseUrls);
  const result = await execa(pythonBin, args, {
    cwd: projectRoot,
    env: process.env,
    all: true,
    reject: false,
  });
  fs.writeFileSync(logFile, result.all ?? '');

  const metrics = parseBenchOutput(result.all ?? '');
  return { exitCode: result.exitCode ?? 0, metrics, logFile };
}

async function runOne({ mode, limiterTarget }: { mode: string; limiterTarget: number | null }) {
  ensureSingleCudaVisible();

  const isDouble = mode === 'double';
  const runLabel = limiterTarget === null ? 'baseline' : `limit_${limiterTarget}`;
  const runDir = resolvePath(resultsRoot, `${mode}_${runLabel}`);
  fs.mkdirSync(runDir, { recursive: true });

  const baseHost = DEFAULT_BASE_HOST;
  const baseUrls = isDouble
    ? [`${baseHost}:8000`, `${baseHost}:8001`]
    : [`${baseHost}:8000`];

  const baseEnv = { ...process.env };
  const runEnv = { ...baseEnv };
  if (limiterTarget !== null) {
    runEnv.HYPERVISOR_IP = '127.0.0.1';
    runEnv.HYPERVISOR_PORT = DEFAULT_HYPERVISOR_PORT;
    runEnv.SHM_PATH = DEFAULT_SHM_PATH;
    runEnv.LD_PRELOAD = limiterSo;
  }

  const metadata = {
    mode,
    limiterTarget,
    baseUrls,
    env: {
      CUDA_VISIBLE_DEVICES: runEnv.CUDA_VISIBLE_DEVICES,
      HYPERVISOR_IP: runEnv.HYPERVISOR_IP ?? null,
      HYPERVISOR_PORT: runEnv.HYPERVISOR_PORT ?? null,
      SHM_PATH: runEnv.SHM_PATH ?? null,
      LD_PRELOAD: runEnv.LD_PRELOAD ?? null,
    },
  };
  fs.writeFileSync(resolvePath(runDir, 'meta.json'), JSON.stringify(metadata, null, 2));

  let hypervisorChild: ReturnType<typeof spawnLogged> | null = null;
  if (limiterTarget !== null) {
    ensureExists('hypervisor binary', hypervisorBin);
    ensureExists('cuda limiter library', limiterSo);

    const hypervisorArgs = [
      'local',
      '--devices',
      '0',
      '--target-util',
      String(limiterTarget),
      '--api-port',
      DEFAULT_HYPERVISOR_PORT,
      '-v',
    ];
    const hypervisorEnv = { ...baseEnv };
    hypervisorChild = spawnLogged(hypervisorBin, hypervisorArgs, {
      cwd: vgpuRoot,
      env: hypervisorEnv,
      logFile: resolvePath(runDir, 'hypervisor.log'),
      name: 'hypervisor',
    });
    registerChild(hypervisorChild);
  }

  const vllmArgs = vllmServeArgs({
    model: runEnv.VLLM_MODEL ?? DEFAULT_MODEL,
    port: '8000',
    gpuMemoryUtilization: isDouble ? DEFAULT_GPU_UTIL_DOUBLE : DEFAULT_GPU_UTIL_SINGLE,
  });

  const vllmChildA = spawnLogged('vllm', vllmArgs, {
    cwd: projectRoot,
    env: runEnv,
    logFile: resolvePath(runDir, 'vllm_8000.log'),
    name: 'vllm:8000',
  });
  registerChild(vllmChildA);

  let vllmChildB: ReturnType<typeof spawnLogged> | null = null;
  if (isDouble) {
    const vllmArgsB = vllmServeArgs({
      model: runEnv.VLLM_MODEL ?? DEFAULT_MODEL,
      port: '8001',
      gpuMemoryUtilization: DEFAULT_GPU_UTIL_DOUBLE,
    });
    vllmChildB = spawnLogged('vllm', vllmArgsB, {
      cwd: projectRoot,
      env: runEnv,
      logFile: resolvePath(runDir, 'vllm_8001.log'),
      name: 'vllm:8001',
    });
    registerChild(vllmChildB);
  }

  await waitForPorts(baseUrls, { timeoutMs: 180_000, intervalMs: 1000 });

  const benchResult = await runBench(runDir, baseUrls);

  killProcessTree(vllmChildA);
  if (vllmChildB) killProcessTree(vllmChildB);
  if (hypervisorChild) killProcessTree(hypervisorChild);

  summaryRows.push({
    mode,
    limiterTarget,
    ...benchResult.metrics,
    benchExitCode: benchResult.exitCode,
  });

  fs.writeFileSync(
    resolvePath(runDir, 'summary.json'),
    JSON.stringify(
      {
        mode,
        limiterTarget,
        bench: benchResult,
      },
      null,
      2,
    ),
  );
}

async function main() {
  if (!skipBaseline) {
    for (const mode of modes) {
      await runOne({ mode, limiterTarget: null });
    }
  }

  for (const util of utils) {
    for (const mode of modes) {
      await runOne({ mode, limiterTarget: util });
    }
  }

  const csvLines = [
    'mode,limiterTarget,totalElapsed,totalTokens,tokensPerSecond,p50Ms,p99Ms,benchExitCode',
  ];

  for (const row of summaryRows) {
    const values = [
      row.mode as string,
      row.limiterTarget ?? '',
      row.totalElapsed ?? '',
      row.totalTokens ?? '',
      row.tokensPerSecond ?? '',
      row.p50Ms ?? '',
      row.p99Ms ?? '',
      row.benchExitCode ?? '',
    ];
    csvLines.push(values.join(','));
  }

  fs.writeFileSync(resolvePath(resultsRoot, 'summary.csv'), csvLines.join('\n'));
  fs.writeFileSync(resolvePath(resultsRoot, 'summary.json'), JSON.stringify(summaryRows, null, 2));

  console.log(`Results saved to ${resultsRoot}`);
}

main().catch(async (err) => {
  console.error('Run failed:', (err as Error).message);
  await cleanup();
  process.exit(1);
});

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
  DEFAULT_PORT_A,
  DEFAULT_PORT_B,
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
  startGpuSampler,
  resolveVllmCommand,
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
  .option('--max-model-len <n>', 'Max model length for vLLM')
  .option('--timeout-s <n>', 'Request timeout seconds', '120')
  .option('--cooldown-s <n>', 'Seconds to wait after each run', '5')
  .option('--hypervisor-bin <path>', 'Hypervisor binary path override')
  .option('--limiter-so <path>', 'libcuda_limiter.so path override')
  .parse(process.argv);

const options = program.opts();

const splitList = (value: string): string[] =>
  value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);

const parseNumberList = (value: string): number[] =>
  splitList(value)
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v));

const toOptionalNumber = (value: unknown): number | undefined => {
  const num = Number(value);
  return Number.isFinite(num) ? num : undefined;
};

const toCsvValue = (value: unknown): string => (value == null ? '' : String(value));

const SERVICE_ACCOUNT_TOKEN_PATH = '/var/run/secrets/kubernetes.io/serviceaccount/token';

const modes = splitList(String(options.modes));
const utils = parseNumberList(String(options.utils));

const skipBaseline = Boolean(options.skipBaseline);
const resultsBase = String(options.resultsDir);
const pythonBin = String(options.python);
const concurrency = Number(options.concurrency);
const totalRequests = Number(options.totalRequests);
const maxTokens = Number(options.maxTokens);
const maxModelLen =
  toOptionalNumber(options.maxModelLen) ?? toOptionalNumber(process.env.VLLM_MAX_MODEL_LEN);
const timeoutSeconds = Number(options.timeoutS);
const cooldownSeconds = Number(options.cooldownS);
const gpuUtilSingle =
  process.env.VLLM_GPU_UTIL_SINGLE ??
  process.env.VLLM_GPU_MEMORY_UTILIZATION ??
  DEFAULT_GPU_UTIL_SINGLE;
const gpuUtilDouble =
  process.env.VLLM_GPU_UTIL_DOUBLE ??
  process.env.VLLM_GPU_MEMORY_UTILIZATION ??
  DEFAULT_GPU_UTIL_DOUBLE;

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

type RunSummary = Record<string, unknown>;
const summaryRows: RunSummary[] = [];
const activeChildren = new Set<ReturnType<typeof spawnLogged>>();

function registerChild(child: ReturnType<typeof spawnLogged> | null) {
  if (!child) return;
  activeChildren.add(child);
}

function stopChild(child: ReturnType<typeof spawnLogged> | null) {
  if (!child) return;
  killProcessTree(child);
  activeChildren.delete(child);
}

async function cleanup() {
  for (const child of Array.from(activeChildren)) {
    killProcessTree(child);
    activeChildren.delete(child);
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
  const args = [
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
  if (Number.isFinite(maxModelLen)) {
    args.push('--max-model-len', String(maxModelLen));
  }
  return args;
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

function limiterTokenAvailable(): boolean {
  try {
    if (!fs.existsSync(SERVICE_ACCOUNT_TOKEN_PATH)) return false;
    const token = fs.readFileSync(SERVICE_ACCOUNT_TOKEN_PATH, 'utf8').trim();
    return token.length > 0;
  } catch {
    return false;
  }
}

function makeRunLabel(limiterTarget: number | null): string {
  return limiterTarget === null ? 'baseline' : `limit_${limiterTarget}`;
}

async function runOne({ mode, limiterTarget }: { mode: string; limiterTarget: number | null }) {
  ensureSingleCudaVisible();

  const isDouble = mode === 'double';
  const runLabel = makeRunLabel(limiterTarget);
  const runId = `${mode}_${runLabel}`;
  const runDir = resolvePath(resultsRoot, `${mode}_${runLabel}`);
  fs.mkdirSync(runDir, { recursive: true });

  const baseHost = DEFAULT_BASE_HOST;
  const baseUrls = isDouble
    ? [`${baseHost}:${DEFAULT_PORT_A}`, `${baseHost}:${DEFAULT_PORT_B}`]
    : [`${baseHost}:${DEFAULT_PORT_A}`];

  const baseEnv = { ...process.env };
  const runEnv = { ...baseEnv };
  const shmPath =
    limiterTarget !== null
      ? resolvePath('/tmp', `tensor-fusion-${nowStamp()}-${runId}`)
      : undefined;

  if (limiterTarget !== null) {
    if (!limiterTokenAvailable()) {
      const benchError = `Limiter disabled: missing service account token at ${SERVICE_ACCOUNT_TOKEN_PATH}`;
      console.error(`[${runId}] ${benchError}`);
      fs.writeFileSync(
        resolvePath(runDir, 'meta.json'),
        JSON.stringify(
          {
            mode,
            limiterTarget,
            baseUrls,
            maxModelLen: maxModelLen ?? null,
            gpuMemoryUtilization: isDouble ? gpuUtilDouble : gpuUtilSingle,
            env: {
              CUDA_VISIBLE_DEVICES: runEnv.CUDA_VISIBLE_DEVICES,
            },
            limiterAvailable: false,
          },
          null,
          2,
        ),
      );
      fs.writeFileSync(
        resolvePath(runDir, 'summary.json'),
        JSON.stringify(
          {
            mode,
            limiterTarget,
            bench: { exitCode: 1, metrics: {}, logFile: resolvePath(runDir, 'bench.log') },
            benchError,
            gpu: null,
          },
          null,
          2,
        ),
      );
      summaryRows.push({
        mode,
        limiterTarget,
        benchExitCode: 1,
        benchError,
        gpuSamples: null,
      });
      return;
    }
    runEnv.HYPERVISOR_IP = '127.0.0.1';
    runEnv.HYPERVISOR_PORT = DEFAULT_HYPERVISOR_PORT;
    runEnv.SHM_PATH = shmPath ?? DEFAULT_SHM_PATH;
    runEnv.LD_PRELOAD = limiterSo;
  }

  const metadata = {
    mode,
    limiterTarget,
    baseUrls,
    maxModelLen: maxModelLen ?? null,
    gpuMemoryUtilization: isDouble ? gpuUtilDouble : gpuUtilSingle,
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
    port: DEFAULT_PORT_A,
    gpuMemoryUtilization: isDouble ? gpuUtilDouble : gpuUtilSingle,
    maxModelLen,
  });
  const { command: vllmCommand, argsPrefix } = resolveVllmCommand();

  const vllmChildA = spawnLogged(vllmCommand, [...argsPrefix, ...vllmArgs], {
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
      port: DEFAULT_PORT_B,
      gpuMemoryUtilization: gpuUtilDouble,
      maxModelLen,
    });
    vllmChildB = spawnLogged(vllmCommand, [...argsPrefix, ...vllmArgsB], {
      cwd: projectRoot,
      env: runEnv,
      logFile: resolvePath(runDir, 'vllm_8001.log'),
      name: 'vllm:8001',
    });
    registerChild(vllmChildB);
  }

  let benchResult = {
    exitCode: 1,
    metrics: {},
    logFile: resolvePath(runDir, 'bench.log'),
  } as Awaited<ReturnType<typeof runBench>>;
  let benchError: string | null = null;
  let gpuStats: Awaited<ReturnType<ReturnType<typeof startGpuSampler>['stop']>> = null;
  let gpuSampler: ReturnType<typeof startGpuSampler> | null = null;

  try {
    await waitForPorts(baseUrls, { timeoutMs: 180_000, intervalMs: 1000 });
    gpuSampler = startGpuSampler(1000);
    benchResult = await runBench(runDir, baseUrls);
  } catch (err) {
    benchError = (err as Error).message;
    console.error(`[${runId}] ${benchError}`);
  } finally {
    if (gpuSampler) {
      gpuStats = await gpuSampler.stop();
    }
    stopChild(vllmChildA);
    if (vllmChildB) stopChild(vllmChildB);
    if (hypervisorChild) stopChild(hypervisorChild);
    if (Number.isFinite(cooldownSeconds) && cooldownSeconds > 0) {
      await new Promise((resolve) => setTimeout(resolve, cooldownSeconds * 1000));
    }
  }

  summaryRows.push({
    mode,
    limiterTarget,
    ...benchResult.metrics,
    benchExitCode: benchResult.exitCode,
    benchError,
    gpuSamples: gpuStats?.samples ?? null,
    gpuAvgUtilGpu: gpuStats?.avg?.utilGpu ?? null,
    gpuAvgUtilMem: gpuStats?.avg?.utilMem ?? null,
    gpuAvgMemUsedMiB: gpuStats?.avg?.memUsedMiB ?? null,
    gpuAvgMemTotalMiB: gpuStats?.avg?.memTotalMiB ?? null,
    gpuAvgPowerW: gpuStats?.avg?.powerW ?? null,
    gpuAvgTempC: gpuStats?.avg?.tempC ?? null,
    gpuMaxUtilGpu: gpuStats?.max?.utilGpu ?? null,
    gpuMaxUtilMem: gpuStats?.max?.utilMem ?? null,
    gpuMaxMemUsedMiB: gpuStats?.max?.memUsedMiB ?? null,
    gpuMaxMemTotalMiB: gpuStats?.max?.memTotalMiB ?? null,
    gpuMaxPowerW: gpuStats?.max?.powerW ?? null,
    gpuMaxTempC: gpuStats?.max?.tempC ?? null,
  });

  fs.writeFileSync(
    resolvePath(runDir, 'summary.json'),
    JSON.stringify(
      {
        mode,
        limiterTarget,
        bench: benchResult,
        benchError,
        gpu: gpuStats,
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
    'mode,limiterTarget,totalElapsed,totalTokens,tokensPerSecond,p50Ms,p99Ms,benchExitCode,gpuSamples,gpuAvgUtilGpu,gpuAvgUtilMem,gpuAvgMemUsedMiB,gpuAvgMemTotalMiB,gpuAvgPowerW,gpuAvgTempC,gpuMaxUtilGpu,gpuMaxUtilMem,gpuMaxMemUsedMiB,gpuMaxMemTotalMiB,gpuMaxPowerW,gpuMaxTempC',
  ];

  for (const row of summaryRows) {
    const values = [
      row.mode as string,
      row.limiterTarget,
      row.totalElapsed,
      row.totalTokens,
      row.tokensPerSecond,
      row.p50Ms,
      row.p99Ms,
      row.benchExitCode,
      row.gpuSamples,
      row.gpuAvgUtilGpu,
      row.gpuAvgUtilMem,
      row.gpuAvgMemUsedMiB,
      row.gpuAvgMemTotalMiB,
      row.gpuAvgPowerW,
      row.gpuAvgTempC,
      row.gpuMaxUtilGpu,
      row.gpuMaxUtilMem,
      row.gpuMaxMemUsedMiB,
      row.gpuMaxMemTotalMiB,
      row.gpuMaxPowerW,
      row.gpuMaxTempC,
    ].map(toCsvValue);
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

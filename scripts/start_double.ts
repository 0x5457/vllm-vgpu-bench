#!/usr/bin/env node
import { execa } from 'execa';
import {
  DEFAULT_GPU_UTIL_DOUBLE,
  DEFAULT_MODEL,
  DEFAULT_PORT_A,
  DEFAULT_PORT_B,
} from './config.js';
import { ensureSingleCudaVisible, resolveVllmCommand, vllmServeArgs } from './lib.js';

try {
  ensureSingleCudaVisible();
} catch (err) {
  console.error(`ERROR: ${(err as Error).message}`);
  process.exit(1);
}

const model = process.env.VLLM_MODEL ?? DEFAULT_MODEL;
const portA = process.env.VLLM_PORT_A ?? DEFAULT_PORT_A;
const portB = process.env.VLLM_PORT_B ?? DEFAULT_PORT_B;
const gpuMemoryUtilization =
  process.env.VLLM_GPU_UTIL_DOUBLE ??
  process.env.VLLM_GPU_MEMORY_UTILIZATION ??
  DEFAULT_GPU_UTIL_DOUBLE;
const maxModelLen = process.env.VLLM_MAX_MODEL_LEN
  ? Number(process.env.VLLM_MAX_MODEL_LEN)
  : undefined;

const argsA = vllmServeArgs({
  model,
  port: portA,
  gpuMemoryUtilization,
  maxModelLen,
});

const argsB = vllmServeArgs({
  model,
  port: portB,
  gpuMemoryUtilization,
  maxModelLen,
});

const { command, argsPrefix } = resolveVllmCommand();
const childA = execa(command, [...argsPrefix, ...argsA], {
  stdio: 'inherit',
  env: process.env,
  reject: false,
});

const childB = execa(command, [...argsPrefix, ...argsB], {
  stdio: 'inherit',
  env: process.env,
  reject: false,
});

const shutdown = (signal: NodeJS.Signals) => {
  childA.kill(signal);
  childB.kill(signal);
};

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

let exitCode: number | null = null;
const handleExit = (code: number | null) => {
  if (exitCode !== null) return;
  exitCode = code ?? 1;
  shutdown('SIGTERM');
  process.exit(exitCode);
};

childA.on('exit', handleExit);
childB.on('exit', handleExit);

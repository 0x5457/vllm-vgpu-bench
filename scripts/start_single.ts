#!/usr/bin/env node
import { execa } from 'execa';
import {
  DEFAULT_GPU_UTIL_SINGLE,
  DEFAULT_MODEL,
  DEFAULT_PORT_SINGLE,
} from './config.js';
import { ensureSingleCudaVisible, vllmServeArgs } from './lib.js';

try {
  ensureSingleCudaVisible();
} catch (err) {
  console.error(`ERROR: ${(err as Error).message}`);
  process.exit(1);
}

const model = process.env.VLLM_MODEL ?? DEFAULT_MODEL;
const port = process.env.VLLM_PORT ?? DEFAULT_PORT_SINGLE;
const gpuMemoryUtilization =
  process.env.VLLM_GPU_MEMORY_UTILIZATION ?? DEFAULT_GPU_UTIL_SINGLE;

const args = vllmServeArgs({
  model,
  port,
  gpuMemoryUtilization,
});

const child = execa('vllm', args, {
  stdio: 'inherit',
  env: process.env,
  reject: false,
});

child.on('exit', (code) => {
  process.exit(code ?? 1);
});

process.on('SIGINT', () => child.kill('SIGINT'));
process.on('SIGTERM', () => child.kill('SIGTERM'));

# LLBuffer AllGather regression

This tests the AllGather implementation in this repository's
`siyshen/low-latency` branch, not an equivalent defect in current upstream
NVIDIA NCCL. The reproduced base is
`5357eff325eddf978137de7140195a5568fa8a11`.

## Bug and fix

`AllGather_LLBufferMC` instantiates the shared implementation with `Unroll=4`.
Previously, the receive phase polled only that one group and wrote ranks
0 through 3. At eight ranks, the upper four output slices were never written.
For non-specialized communicators smaller than four ranks, an unconditional
four-peer receive can instead poll nonexistent peers. Checking `r < nRanks`
**after** that receive does not make the polling safe.

Two byte-layout problems occur in the same loop:

- Output slices used `ceil(bytes / 8) * 8` as their stride, although AllGather
  requires tightly packed, unpadded slices of exactly `bytes` bytes.
- The generic packed-load helper's byte-array path can access outside an
  unaligned or partial input pack and leave padding uninitialized.

The fix receives all complete peer groups followed by a guarded partial group.
It uses exact byte strides, keeps an aligned full-pack load, and assembles
partial/unaligned packs with bounded byte loads and initialized padding.
Pack indexing uses `size_t`; only the bounded 1..8-byte tail size is narrowed.
The synchronization protocol, public API and dispatch policy are unchanged.

## Build

Requirements: Linux, a C++14-capable host compiler, CUDA, and this branch's
NCCL library. No MPI, Python, PyTorch or SGLang is required by the test.
Multicast requires suitable NVLink/NVSwitch hardware; B300 validation uses
SM103. Substitute the appropriate CUDA architecture for other hardware.

From the repository root:

```sh
make -j24 src.build NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103"
make -f test/unit/Makefile.all_gather CUDA_ARCH=sm_103
```

The default executable is `build/all_gather_llbuffer_test`. `NCCL_HOME`,
`CUDA_HOME`, `CUDA_ARCH` and `TEST_BINARY` can be overridden. Use separate
build directories/worktrees for before/after libraries, and verify the loaded
library with `ldd` or an equivalent loader inspection. A shared version number
alone does not distinguish the original and repaired binaries.

## Minimal reproducer

Run on eight otherwise idle GPUs. These environment settings apply only to
the test process; no host configuration changes are necessary.

```sh
export LD_LIBRARY_PATH="$PWD/build/lib:/usr/local/cuda/lib64"
export NCCL_CUMEM_ENABLE=1 NCCL_WIN_ENABLE=1 NCCL_NVLS_ENABLE=1
export NCCL_GIN_ENABLE=0 NCCL_GRAPH_MIXING_SUPPORT=1
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo
export NCCL_SYM_KERNEL=AllGather_LLBufferMC NCCL_SYM_LLBUFFER_SYNC=0
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING

timeout --signal=TERM --kill-after=10 90 \
  ./build/all_gather_llbuffer_test --ranks 8 --bytes 8 --eager 1 --graphs 0
```

To test the unmodified base, build its library in a separate worktree and
compile this same test against those headers/libraries. The harness itself
uses only the public API, so no library fix is necessary to build the
reproducer.

Expected before the fix: exit status 1, with this failure on each rank
(concurrent output order is not significant):

```text
FAIL rank=0 ... index=160 actual=201 expected=32 mismatches=32
```

Index 160 is the 128-byte leading guard plus 32 bytes: the start of rank 4's
slice. Value 201 is the unwritten-output canary. The eight-byte payloads
should occupy all 64 output bytes, not just the lower 32.

Expected after the fix: exit status 0, eight `PASS rank=... checks=1` lines
and `ALL_PASS ranks=8 total_checks=8`.

## Do not accept a fallback as a pass

The test forks **before any CUDA or NCCL API call**, then uses one process per
GPU. This matters: a multi-thread/multi-GPU single-process setup can take a
different path in this branch and pass without exercising the broken kernel.

Confirm the actual enqueue log contains:

```text
AllGather [Symmetric]: 8 Bytes -> Kernel AllGather_LLBufferMC
```

Reject `AllGather: ... -> Algo RING ...` as evidence for this regression.
The separate `SymKernel: ... nElts=1024` initialization/tuning messages alone
do **not** prove the user's gather executed that kernel. Use
`NCCL_DEBUG_FILE=.../nccl-%p.log` for separate process logs if output interleaves.
Those raw debug logs can contain hostnames and addresses; redact them before
sharing. A GPU kernel trace provides an additional dispatch check.

## Extended correctness suite

With the environment above, run:

```sh
for ranks in 2 3 4 5 8; do
  timeout --signal=TERM --kill-after=10 120 \
    ./build/all_gather_llbuffer_test --ranks "$ranks"
  timeout --signal=TERM --kill-after=10 120 \
    ./build/all_gather_llbuffer_test --ranks "$ranks" --in-place
done
```

Repeat with `NCCL_SYM_KERNEL=AllGather_LLBuffer`, first with
`NCCL_SYM_LLBUFFER_SYNC=0` and then `NCCL_SYM_LLBUFFER_SYNC=1`. Check actual
enqueue dispatch for these modes too; rank-specialized names are expected
where applicable. The MC entry point tested here uses poison synchronization.

The default suite covers 11 byte-count/offset pairs:
`8/0, 1/0, 2/1, 7/3, 9/5, 14/1, 42/2, 40960/0, 40962/7, 327680/32, 2621442/3`.
Each has three validated eager calls before capture. One mixed-size CUDA
Graph is then replayed 16 times, queued without CPU/rank barriers between
launches. This is **209 checked calls per rank** per invocation.

A GPU producer increments the epoch and regenerates rank-dependent data on
every call. Counter initialization, producers and verifiers use the same
nonblocking CUDA stream. The final rank is delayed by default. A GPU consumer
checks every output byte, unchanged separate input data, and leading/trailing guards
before reuse. Failures accumulate across all graph replays rather than being
cleared by a later passing call. The final epoch also verifies that the
requested number of operations actually executed.

Use `--graphs 512 --bytes 8` for a longer same-shape queued-replay check.
The whole command must remain externally time-bounded because a broken
device polling loop need not return an error to the host.

## Validation and limitations

Validation hardware: eight NVIDIA B300 SXM6 GPUs in one NVLink/NVSwitch domain,
NV18 all-pairs topology, 275040 MiB reported per GPU. Software: CUDA 13.0.88,
driver 580.126.09, NCCL 2.29.2 from the base named above plus this fix.
The standalone test builds with `-Wall -Wextra` without warnings. Both library
builds emitted the same 191 existing warnings, with no new warning diagnostics;
no blanket warning-free library build is claimed.

Completed MC poison checks on the same hardware and software, all with
`--eager 1 --graphs 0` and the following `--ranks`, `--bytes` and `--offset`:

| Case | Ranks | Bytes | Offset | Original base | Fixed library |
| --- | --- | --- | --- | --- | --- |
| Missing upper ranks | 8 | 8 | 0 | Fail, exit 1 | Pass, exit 0 |
| Non-pack-sized stride | 4 | 14 | 0 | Fail, exit 1 | Pass, exit 0 |
| Unaligned input | 4 | 8 | 1 | Fail, exit 1 | Pass, exit 0 |

All three original cases reported byte mismatches on every rank; the same
test binary passed against the repaired library. Actual enqueue logs confirmed
`AllGather_LLBufferMC` in both variants, not a fallback.

The fixed-library matrix passed all **30 configurations**: multicast poison,
unicast poison and unicast LL16, each with 2, 3, 4, 5 and 8 ranks, both in-place
and out-of-place. This totals **27,588 checked rank-calls**, including eager
and queued mixed-size graph calls. Every configuration's actual enqueue logs
showed the requested kernel family/synchronization mode with no Ring fallback.
The eight-rank `--bytes 8 --graphs 512` stress test passed another 4,120 checked
rank-calls.

An eight-rank default-suite Nsight Systems trace, using CUDA graph **node**
tracing, recorded 1,672 executions of
`ncclSymkDevKernel_AllGather_LLBufferMC`: 209 on each of eight GPUs/processes.
The default graph-level trace alone does not expose individual replay kernels.

Compute Sanitizer memcheck (`--target-processes all --error-exitcode 99`,
test arguments `--ranks 8 --graphs 2`) completed all 440 checked rank-calls,
but exited **99**, reporting 48 `CUDA_ERROR_NOT_PERMITTED` errors from
`cuMemCreate` in `ncclMemAlloc` at `src/allocator.cc:59`. This is the existing
FABRIC-handle probe: the allocator explicitly handles this result and retries
with POSIX handles. The log reported no device memory-access errors, but this
is **not a clean sanitizer pass**. No API-error suppression or host-permission
changes were used.

This is a correctness test, **not a latency benchmark or performance claim**.
It deliberately includes producer, verifier, guard and host synchronization
work. Poison-mode payloads exclude reserved sentinel bit patterns; this fix
does not remove that protocol restriction. Input/output guards do not by
themselves prove absence of all out-of-bounds reads. More-than-eight-rank,
multi-node and other GPU-architecture coverage are not established here.
This defect is separate from any model-level numerical failure.

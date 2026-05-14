# DISSCO MPI Per-Sound Parallelization Guide

## Purpose

This document explains how the current MPI parallelization in DISSCO 2.1.0 works.
It is intended to be a code-level guide for developers who want to:

- understand the design choices behind the MPI implementation
- trace the runtime path from `main()` to final audio output
- debug correctness problems such as rank divergence or reduction issues
- extend the current implementation into a more scalable future design

This guide describes the **current implementation**, not the older experimental
MPI-per-partial approach.

## Summary

The current MPI design is **per-sound**, not per-partial.

At a high level:

1. All MPI ranks parse the same `.dissco` file.
2. All MPI ranks build the same CMOD event tree and create the same sequence of `Sound` objects.
3. Each `Sound` is assigned a deterministic ordinal `k`.
4. Ownership is assigned by round robin: `rank = k % mpi_size`.
5. Only the owning rank keeps and renders that `Sound`.
6. Each rank mixes the sounds it owns into a local score buffer.
7. At the end, rank 0 gathers the final score by reducing all local score buffers.
8. Only rank 0 applies score-level post-processing and writes the output file.

This means DISSCO currently uses:

- **replicated composition generation**
- **distributed sound rendering**
- **root-only final score assembly**

## Why This Design Exists

The original hotspot in DISSCO is in LASS rendering, especially:

- `Sound::render()`
- `Partial::render()`
- score mixing

CMOD event generation is recursive, stateful, and tightly coupled to random
decisions and DOM traversal. That makes it difficult to distribute safely
without a much larger redesign.

The per-sound MPI design therefore takes a narrower approach:

- leave CMOD generation replicated on every rank
- split the independent `Sound` rendering work across ranks
- reduce the final mixed buffers only once, at the end

Compared with the earlier per-partial experiment, this design:

- avoids MPI collectives inside every single sound render
- distributes sound-level filter/reverb/spatialization as part of the owned sound
- communicates once per score instead of once per sound or partial group

## Build-Time Activation

MPI is enabled through the premake option:

```bash
premake4 --mpi
```

The `--mpi` option defines `USE_MPI` in multiple targets in `premake4.lua`:

- `lass`
- `lcmod`
- `cmod`

Important note: the premake option text in `premake4.lua` still says:

> "Enable MPI-based partial rendering in LASS Sound::render"

That description is stale. The implementation is now **MPI per-sound**.

Typical build sequence:

```bash
cd DISSCO-2.1.0
premake4 --mpi
make cmod config=debug CC=mpicc CXX=mpic++
```

Representative run commands:

```bash
mpirun -n 4 ./cmod /path/to/project.dissco
```

or on Slurm systems:

```bash
srun -n 4 ./cmod /path/to/project.dissco
```

Cluster-specific MPI launch flags depend on the site runtime.

## MPI Helper Layer

MPI support is centralized in:

- `LASS/src/MPIWrapper.h`

This wrapper provides:

- initialization through `dissco_mpi::ensureInitialized()`
- `rank()`, `size()`, and `isRoot()`
- `barrier()`
- `broadcastInt()` and `broadcastString()`
- `localRenderThreads()`

### Thread Support Choice

`ensureInitialized()` requests:

```cpp
MPI_THREAD_SERIALIZED
```

This is important because DISSCO still uses pthreads internally. The current MPI
path is designed so that MPI calls happen in controlled places rather than from
multiple worker threads simultaneously.

## High-Level Runtime Flow

The actual runtime path is:

1. `CMOD/src/Main.cpp`
2. `CMOD/src/piece-experimental.cpp`
3. `CMOD/src/Event.cpp`
4. `CMOD/src/Bottom.cpp`
5. `CMOD/src/Utilities.cpp`
6. `LASS/src/Score.cpp`
7. `LASS/src/Sound.cpp`
8. `LASS/src/Partial.cpp`

Each stage is described below.

## Stage 1: Process Startup in `Main.cpp`

`CMOD/src/Main.cpp` is the entry point.

### What happens

1. If `USE_MPI` is defined, `dissco_mpi::ensureInitialized()` is called.
2. A SIGSEGV handler is installed.
3. Non-root ranks silence `std::cout`.
4. Rank 0 creates output directories.
5. All ranks synchronize with a barrier.
6. Every rank constructs the same `Piece`.

### Why non-root output is silenced

Without this, every rank would print the same CMOD progress messages, which
would make logs unreadable. Root is treated as the user-facing rank.

### Why rank 0 creates directories

All ranks share the same project path. Directory creation is root-only to avoid
 redundant filesystem races.

## Stage 2: Piece Setup in `piece-experimental.cpp`

`Piece::Piece()` is where DISSCO parses the `.dissco` file and sets up the run.

### What happens

1. MPI is initialized again defensively if needed.
2. The `.dissco` XML file is parsed with Xerces.
3. Global configuration is read:
  - title
  - file flags
  - piece duration
  - channel count
  - sample rate
  - sample size
  - configured thread count
4. If MPI is active with more than one rank, local render threads are forced to 1.
5. Rank 0 obtains the random seed and `numRuns`, then broadcasts them.
6. Every rank seeds the same CMOD random stream.
7. A `Utilities` object is created.
8. The top event is created and `buildChildren()` is called.

### Why local threads are forced to 1

The current MPI design uses rank-level parallelism as the main source of
distributed work. Forcing local render threads to 1 simplifies correctness and
avoids nested oversubscription while the MPI path is still being stabilized.

This does **not** remove the internal worker/composite pipeline. It means each
rank will typically have:

- one producer thread building CMOD sounds
- one render worker thread
- one composite thread

### Why the seed is broadcast

Because CMOD generation is still replicated on all ranks, every rank must make
the same random choices while building the event tree. If the seed differs,
the entire ownership model breaks.

## Stage 3: Recursive CMOD Event Generation

The composition tree is expanded through:

- `CMOD/src/Event.cpp`
- `CMOD/src/Bottom.cpp`

### `Event::buildChildren()`

Each event:

1. decides how to place children
2. creates child events
3. recursively calls `buildChildren()` on those children

This recursion is still fully replicated across every MPI rank.

### Where sounds are created

Leaf-level sound generation happens in:

- `Bottom::buildSound()`

This function:

1. creates a `Sound`
2. sets `START_TIME` and `DURATION`
3. computes base frequency and loudness
4. creates the partial list
5. applies sound modifiers
6. applies spatialization, filter, and reverb setup
7. hands the completed `Sound` to `Utilities::addSound()`

At this point the sound is fully defined as an object and is ready to be
scheduled for rendering.

## Stage 4: The CMOD-to-LASS Handoff

The bridge between composition and rendering is:

- `CMOD/src/Utilities.cpp`

### `Utilities::addSound()`

This is the exact handoff point from CMOD to LASS.

- If sound synthesis is disabled, the sound is discarded.
- Otherwise the `Sound`* is passed to `score->add(_sound)`.

### `Utilities::doneCMOD()`

When event generation is finished, CMOD calls:

```cpp
score->doneAddingSounds()
```

That tells the rendering side there will be no more sounds.

## Stage 5: Score Ownership and Scheduling in `Score.cpp`

The MPI ownership logic lives in:

- `LASS/src/Score.cpp`
- `LASS/src/Score.h`

This is the core of the per-sound parallelization.

### The local Score pipeline still exists

`Score` still manages the older pthread-based producer/consumer design:

- a producer side that adds `Sound*` objects
- worker thread(s) that render sounds
- a composite thread that mixes rendered sounds into the local score

MPI did not replace that pipeline. It changed **which sounds** each rank keeps.

### `Score::add(Sound* _sound)`

This function now does four important things:

1. assign a deterministic `soundOrdinal`
2. validate that all ranks generated the same sound at that ordinal
3. decide ownership
4. queue only owned sounds for local rendering

#### Step 1: Ordinal assignment

The next ordinal is tracked by:

```cpp
nextSoundOrdinal
```

All ranks increment this in the same order because they are all running the same
replicated CMOD construction path.

#### Step 2: Consistency validation

In debug builds, `validateSoundConsistency()` computes a lightweight signature
using:

- sound ordinal
- `START_TIME`
- `DURATION`
- partial count
- total duration

All ranks compare min and max signature values with `MPI_Allreduce`.

If they differ, MPI aborts.

This check exists because the ownership model assumes:

> sound ordinal `k` refers to the same logical sound on every rank

If ranks drift apart, assigning `k % mpi_size` becomes meaningless.

#### Step 3: Ownership decision

Ownership is determined by:

```cpp
soundOrdinal % mpi_size == mpi_rank
```

This is implemented by `Score::ownsSound()`.

#### Step 4: Non-owners discard

If a rank does not own a sound:

- it deletes the sound immediately
- that sound never enters the local render queue

If it does own the sound:

- it is pushed into the local `sounds` queue
- `scoreEndTime` is updated from that owned sound

### Why ownership is assigned here

This location is a practical boundary:

- CMOD has already fully defined the sound
- rendering has not started yet
- the sound is now self-contained enough to hand off to a rank

That makes `Score::add()` the cleanest place to switch from replicated work to
distributed work.

## Stage 6: Local Rendering on the Owning Rank

Once a sound is owned locally, the existing worker/composite pipeline takes over.

### Worker thread entry point

The render worker thread pops sounds from the queue and calls:

```cpp
sound->render(numChannels, samplingRate)
```

### Composite thread role

Rendered `MultiTrack` objects are handed back to the score with:

```cpp
addRenderedSound(startTime, renderedSound)
```

The composite thread then mixes each rendered sound into the rank-local
`scoreMultiTrack` buffer at the correct start time.

### Important consequence

Each MPI rank ends up with:

- a **complete local mix**
- but only for the sounds it owns

This local score is the object later sent into the final MPI reduction.

## Stage 7: Whole-Sound Rendering in `Sound::render()`

The old MPI-per-partial logic is no longer the active design.

`Sound::render()` is now local again.

### What `Sound::render()` does

For a single owned sound it:

1. calculates loudness
2. sets up detuning envelopes if needed
3. renders each partial
4. composites all partial results into one `MultiTrack`
5. applies sound-level filter
6. applies sound-level reverb
7. applies sound-level spatialization

### Why this matters

This means the unit of MPI work is the **entire sound**, not the partial.

So when a rank owns a sound, it owns:

- all of its partial computation
- all sound-level DSP
- its contribution to the local score mix

This is one of the biggest conceptual differences from the older per-partial
approach.

## Stage 8: Deterministic Partial Rendering

One subtle but critical fix lives in:

- `LASS/src/Partial.cpp`

### The problem

CMOD generation still uses a global random stream. Earlier render-side code
used `rand()`/`srand()` inside rendering, which could perturb that global state
while CMOD was still constructing later sounds.

That caused rank divergence.

### The current fix

`Partial::render()` now uses a **local deterministic generator** derived from:

- partial number
- frequency
- relative amplitude
- wave shape
- loudness scalar
- sample count
- sampling rate
- duration

The render loop then uses that local generator for transient decisions and
other render-time random behavior.

### Why this is required

Because the design depends on all ranks generating the same `Sound` sequence
before ownership filtering. Any render-side mutation of CMOD's random state can
break that assumption.

## Stage 9: Finalization in `Score::doneAddingSounds()`

When CMOD finishes building sounds, it calls:

```cpp
Score::doneAddingSounds()
```

This does the following:

1. marks the producer side as finished
2. unblocks the worker thread(s)
3. joins the render worker thread(s)
4. joins the composite thread
5. chooses the finalization path

If MPI is not active, the score is finalized locally.

If MPI is active with multiple ranks, the code enters:

```cpp
reduceScoreToRoot()
```

## Stage 10: Final Score Reduction in `reduceScoreToRoot()`

This function is the final distributed step.

### Step 1: agree on score length

Each rank has its own `scoreEndTime`, based only on the sounds it owned.

Ranks compute the global maximum with:

```cpp
MPI_Allreduce(..., MPI_MAX)
```

Then each rank resizes its local score buffer to that shared length.

This is required so every rank contributes arrays with the same shape.

### Step 2: gather ownership counts

Each rank reports how many sounds it owned with:

```cpp
MPI_Gather
```

Rank 0 logs a summary such as:

```text
MPI per-sound ownership: rank 0=... rank 1=... ... (total sounds=...)
```

This is a debugging and load-balance visibility aid.

### Step 3: flatten the local score

Each rank converts its local `MultiTrack` into two contiguous buffers:

- wave samples
- amplitude samples

Both are needed because DISSCO tracks both signal and amplitude information.

### Step 4: reduce to rank 0

The flattened wave buffer and amp buffer are separately reduced with:

```cpp
MPI_Reduce(..., MPI_SUM, root=0)
```

Why summation is correct:

- each rank owns a disjoint subset of sounds
- score mixing is additive
- the full score is therefore the sum of all local score contributions

### Step 5: non-root ranks return `NULL`

After contributing their local buffers:

- non-root ranks delete their local `scoreMultiTrack`
- they return `NULL`

They do not own the final score object.

### Step 6: rank 0 reconstructs the final score

Rank 0 rebuilds a new `MultiTrack` from the reduced wave and amp buffers.

Only after that does rank 0 run score-level post-processing.

## Stage 11: Root-Only Score-Level Post-Processing

After reduction, rank 0 alone performs:

- score-level reverb
- clipping management

This happens in `reduceScoreToRoot()` after the final `MultiTrack` has been
reconstructed.

### Why this is root-only

These operations are score-global and potentially nonlinear.

If each rank applied them independently before reduction, the sum would not
necessarily equal the correct final result.

So the design intentionally waits until the global score exists on rank 0.

## Stage 12: Final Audio Output

Back in `Piece::Piece()`:

- `utilities->doneCMOD()` returns a `MultiTrack*`
- in MPI mode, only rank 0 receives a real pointer
- non-root ranks receive `NULL`

Rank 0 writes the audio file with `AuWriter::write(...)`.

This root-only write avoids duplicate output files and matches the fact that
rank 0 alone owns the final reduced score.

## Responsibilities by Rank

### Work done by all ranks

- initialize MPI
- parse the `.dissco` file
- build the same CMOD event tree
- create the same candidate `Sound` objects
- assign sound ordinals in the same order
- perform debug consistency checks
- render and mix locally owned sounds
- participate in the final score reduction

### Work done only by owner ranks for a specific sound

- keep the sound object
- render the sound
- add the rendered sound into the local score

### Work done only by rank 0

- create output directories
- interact with the user for seed and `numRuns`
- receive the final reduced score
- apply score-level post-processing
- write audio output
- print the ownership summary

### Work done by non-owner ranks for a specific sound

- discard that sound immediately in `Score::add()`

## What Is Parallelized and What Is Not

### Parallelized

- whole-sound rendering across ranks
- local score mixing for owned sounds
- final distributed reduction of local score buffers

### Not parallelized across ranks

- CMOD event generation
- XML parsing
- sound construction up to the `Score::add()` boundary

### Still local inside a sound

- per-partial rendering
- per-sound filter/reverb/spatialization

## Current Limitations

The implementation works, but it still has important limitations.

### 1. CMOD is replicated on every rank

Every rank parses and builds the whole piece. This increases:

- memory use
- XML parsing overhead
- event-generation overhead

MPI only reduces rendering work, not composition-generation work.

### 2. Load balance depends on sound granularity

Round-robin ownership is simple and deterministic, but it assumes sound cost is
roughly balanced.

If one sound is much heavier than another, rank balance may degrade.

### 3. Debug divergence checking is debug-only

`validateSoundConsistency()` is compiled only when assertions/debug behavior is
enabled. In release builds, that safety net is absent.

### 4. Final reduction is centralized

Rank 0 still reconstructs and post-processes the full score, so the final stage
is not distributed.

### 5. Local render threads are intentionally limited

The current implementation forces local render threads to 1 in MPI mode. That
keeps the design simple, but it also means hybrid MPI + shared-memory scaling
is not yet being fully exploited.

## Why This Replaced the Older Per-Partial Idea

The earlier MPI direction attempted to parallelize at the partial level inside
`Sound::render()`.

That approach had several downsides:

- communication inside the sound-render hot path
- more frequent collectives
- duplication of later sound-level post-processing on every rank
- greater sensitivity to per-sound buffer shape mismatches

The per-sound design is easier to reason about because:

- ownership is defined once per sound
- rendering stays local after ownership is assigned
- MPI communication happens at well-defined boundaries

## Big Picture Pseudocode

```text
initialize MPI

all ranks parse the same project
rank 0 gets seed and numRuns
broadcast seed and numRuns
all ranks seed CMOD RNG identically

for each generated Sound in the replicated CMOD stream:
    soundOrdinal++
    debug-check that all ranks built the same sound
    owner = soundOrdinal % mpi_size
    if this rank is not owner:
        delete sound
    else:
        queue sound for local rendering

local worker renders owned sounds
local composite thread mixes them into a rank-local score

when no more sounds remain:
    join local worker/composite threads
    allreduce max score length
    flatten local score buffers
    reduce wave buffer to rank 0
    reduce amp buffer to rank 0

if rank == 0:
    rebuild final score
    apply score-level reverb and clipping
    write output
else:
    return NULL
```

## File Map for Future Maintenance

If you need to work on this implementation, start here:

- `CMOD/src/Main.cpp`
  - process startup and root-only filesystem setup
- `CMOD/src/piece-experimental.cpp`
  - `.dissco` parsing, seed broadcast, thread forcing, root-only output
- `CMOD/src/Event.cpp`
  - recursive event construction
- `CMOD/src/Bottom.cpp`
  - actual `Sound` object creation
- `CMOD/src/Utilities.cpp`
  - handoff from CMOD to `Score`
- `LASS/src/MPIWrapper.h`
  - MPI helper abstraction
- `LASS/src/Score.cpp`
  - ownership, local scheduling, local mix, final MPI reduction
- `LASS/src/Sound.cpp`
  - whole-sound render path on the owner rank
- `LASS/src/Partial.cpp`
  - deterministic local render-time RNG

## Recommended Future Extensions

If this design is extended, the most likely next steps are:

1. distribute CMOD generation instead of replicating it
2. reintroduce controlled intra-rank threading on top of MPI ownership
3. improve load balancing beyond strict round robin
4. reduce root bottlenecks in final post-processing
5. make divergence validation available in selected release/debug configurations


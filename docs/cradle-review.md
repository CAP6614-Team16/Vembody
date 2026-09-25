# Cradle implementation review for Vembody

Reviewed on 2026-09-10 against commit [`d7752fccf890d8d3818cd1d435f3705f604a1339`](https://github.com/BAAI-Agents/Cradle/commit/d7752fccf890d8d3818cd1d435f3705f604a1339), dated 2024-11-07. The GitHub API resolved `main` to this commit during review. Links below are pinned so future upstream changes do not change the evidence.

This is a targeted static source review, not a reproduction of Cradle's experiments or a complete audit. Inspected the app and Stardew runners, planning and reflection prompts, skill parsing/registration, input and capture utilities, memory, model-provider code, configuration, dependencies, and license. Downloaded selected files to a temporary directory; did not install Cradle or execute desktop actions.

Recommendation: retain Vembody's minimal baseline plan. Cradle supplies useful mechanisms for observation, feedback, and memory, but its implementation also shows why execution safety and independent evaluation must be designed explicitly.

## What Cradle does

The app runner sequences information gathering, reflection, task inference, skill curation, action planning, and execution. It saves memory and periodic checkpoints. The app's skill-curation method is a placeholder; the existence of a stage does not mean it performs model work in every runner. Reflection can receive before/after images, previous actions, execution errors, history, and image-change signals. [App runner](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/runner/app_runner.py#L182).

The execution provider performs selected skills, captures the resulting screen, and records execution information and frame boundaries. Game pausing is part of this path. [Execution provider](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/provider/execute/skill_execute.py#L30).

For Vembody, keep one action proposal per observation initially. Store the resulting observation and execution status even before implementing verification. This makes later memory and verification experiments possible without changing the baseline's evidence collection.

## Lessons that should affect implementation

### 1. Stopping is not proof of success

The app runner normalizes missing or empty actions into an empty action representation. Repeated empty actions can stop the episode and produce a success message. This is a termination heuristic, not an independent task check. [Action postprocessing](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/runner/app_runner.py#L768).

Vembody should record `termination_reason` separately from `evaluation_status`. A model's `finish` action requests evaluation. Only a predefined external check or human rubric establishes success. Empty output, malformed output, timeout, and exhausted action budgets must not count as successful completion.

### 2. Input cleanup needs direct tests

Two concrete source-level issues appear in `IOEnvironment`: `release_held_keys()` removes tracking entries without issuing `key_up`; `update_timeouts()` returns when no keys are held, bypassing its later mouse-button timeout processing. These are static findings; their end-to-end effects were not reproduced. [Key cleanup](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/gameio/io_env.py#L484), [timeout processing](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/gameio/io_env.py#L175).

The Windows backend also explicitly disables PyDirectInput's fail-safe. [Backend setup](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/gameio/gui_utils.py#L35).

Vembody should initially avoid holds across model calls. Pair input-down operations with release in `finally`, retain enough state for emergency cleanup, and test keyboard-only and mouse-only cleanup independently. Keep an emergency stop usable during inference; an action counter alone cannot enforce a wall-clock deadline while a model request stalls.

### 3. Window matching must enforce the intended boundary

Cradle's active-window helper accepts certain generic dialog titles and can accept a matching process name. These accommodate application workflows, but do not uniquely identify the selected target. [Window checks](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/gameio/lifecycle/ui_control.py#L81).

For Vembody, validate the approved window identity and capture geometry immediately before input. If focus, position, or dimensions changed during inference, stop or obtain a fresh observation. Any supported child dialog needs an explicit ownership policy. A capture rectangle limits what is seen; by itself it does not restrict where keyboard input goes.

### 4. Parse actions as data

Cradle parses function-call expressions with Python AST utilities and literal argument evaluation. Separately, its skill-registration path executes supplied skill source with `exec`; stored skills can also be reconstructed from source. These are distinct mechanisms, so it would be inaccurate to describe every selected action as arbitrary code execution. [Expression parser](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/environment/skill_registry.py#L206), [skill registration](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/environment/skill_registry.py#L362).

Vembody's smaller contract should be strict JSON mapped to a typed, allowlisted action. Reject unknown fields, invalid values, nonfinite coordinates, excessive durations, and disallowed keys. Do not add dynamic skill-code execution. A parser failure must produce a logged failure or bounded retry with no live input.

### 5. Memory should preserve aligned events

`LocalMemory` keeps bounded histories per field, a working dictionary, and a replaceable summary. It is a singleton, and persistence saves recent history rather than the entire working state. [Local memory](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/memory/local_memory.py).

Borrow bounded history and summaries. For Vembody, store one complete step record containing the observation, proposed action, actual execution, and outcome. Give each episode its own memory instance so trials cannot share state accidentally. Keep the append-only experiment trace separate from the short history sent to the model. A saved history should not be presented as a resumable checkpoint unless the environment and runtime state can also be restored.

### 6. Visual change is a signal, not a verdict

Cradle passes image-change and mouse-position signals into reflection. Its Chrome reflection prompt asks the model to judge both the previous action and the overall task. [Reflection inputs](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/runner/app_runner.py#L460), [reflection prompt](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/res/chrome/prompts/templates/self_reflection.prompt).

For Vembody, preserve three separate facts: input was delivered, a visible change occurred, and the expected outcome was verified. Animation can change an image after an ineffective click; a valid keyboard focus change can be visually subtle. Support an uncertain outcome, and keep model reflection separate from the benchmark's final judge.

### 7. Local inference changes the cost of the architecture

Cradle's provider implements remote chat completions, bounded API retries, and returned token-usage information. Its Chrome planning prompt includes history, visual descriptions, a skill library, and detailed reasoning instructions. [Model provider](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/provider/llm/openai.py#L293), [planning prompt](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/res/chrome/prompts/templates/action_planning.prompt).

Inference: copying this multi-stage design may make local interaction too slow; that must be measured on team hardware. Begin with a compact prompt, one model call per proposed action, and a concise rationale. Measure every call and retry, including future summary and verification calls. Set request deadlines from the remaining episode budget. Do not assume a local model can interpret multiple images or follow these prompts equally well without saved-screenshot trials.

### 8. Grounding and environment setup remain experimental variables

The Chrome prompt prefers labeled visual regions where available and contains application-specific heuristics. The capture utility records a configured region; input conversion uses configured geometry. [Planning prompt](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/res/chrome/prompts/templates/action_planning.prompt), [capture](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/gameio/lifecycle/ui_control.py#L165), [coordinate conversion](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/cradle/gameio/io_env.py#L269).

For Vembody, test corners, nonzero capture offsets, image resizing, and Windows display scaling before interpreting missed clicks as model failures. Define normalized endpoints so `1.0` lands on the last valid pixel. If visual labels are added later, bind their mapping to the exact screenshot used for the proposal. Treat OCR, labels, and handcrafted hints as explicit configuration changes in comparisons.

## What to carry into Milestone 1

- One typed step record and JSONL logging, including raw model output and all failure paths.
- Mock inference and dry-run execution by default; no model or final environment selection yet.
- Strict action validation and exactly one action per iteration.
- Explicit action and elapsed-time limits, bounded request timeouts, and guaranteed input cleanup.
- Tests for malformed responses, coordinate boundaries, loss of focus, interrupted execution, and episode termination. Test desktop behavior through a fake input backend first.
- Separate agent termination from externally evaluated success.
- A minimal provider interface and dependency injection; memory and reflection remain later experiments.

These are recommendations for the upcoming implementation, not claims that the features already exist. At review time Vembody contains `README.md`, `.gitignore`, and the project guide, with no agent implementation.

## Reuse and remaining evidence

Cradle's repository uses the MIT license and requires preservation of its copyright and permission notice when copying substantial portions. No upstream code was added to Vembody in this review. Its dependency list includes substantial perception tooling and older version pins; use it to identify components, then select and verify only dependencies needed by implemented Vembody features. [License](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/LICENSE), [dependencies](https://github.com/BAAI-Agents/Cradle/blob/d7752fccf890d8d3818cd1d435f3705f604a1339/requirements.txt).

This inspection supports engineering decisions, not performance claims or research novelty. Before choosing the final contribution, review the related papers and their ablations, establish Vembody's local baseline, and collect repeated trials. Memory and reflection are already present in Cradle; any proposed contribution needs a more precise hypothesis and controlled evidence.

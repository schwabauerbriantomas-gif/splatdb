# Claurst Parallel Protocol

## Principle: NARROW + PARALLEL + PRE-FED CONTEXT

Each Claurst instance gets:
1. Max 2-3 related files to edit
2. Exact instructions (not "read the plan" — give the fixes inline)
3. Pre-read file snippets embedded in the prompt
4. One compilation check at the end
5. Max 20 turns

## Steps

### 1. Partition work into independent units
Group fixes by file dependency. Units that touch different files can run in parallel.
Max 3 units simultaneously (resource limit).

### 2. Pre-read and embed context
Before launching, read each target file and include the relevant sections
in the prompt. This saves Claurst from wasting turns reading.

### 3. Launch parallel instances
Each instance gets a self-contained prompt with:
- Exact file paths
- Exact line numbers and current code
- Exact replacement code
- "Edit ONLY these files. Do NOT read other files."
- "Run cargo check at the end."

### 4. Monitor and collect
Poll all instances. When one finishes, verify its changes.

### 5. Compile checkpoint
After each batch of 3, run cargo check. Fix any conflicts.

### 6. Fallback rule
If a Claurst instance fails or produces garbage 2 times on the same unit,
do it myself with patch tool.

## Unit sizing heuristic
- 1-3 files that import each other = 1 unit
- If a unit needs >5 distinct edits, split it
- Never put more than 8 edits in one prompt

## When to use Claurst vs myself
- Claurst: mechanical edits (replace unsafe, add validations, add constants)
- Myself: architectural changes (new middleware, new types, cross-file refactors)
- Mixed: I do architecture, Claurst fills in the mechanical parts

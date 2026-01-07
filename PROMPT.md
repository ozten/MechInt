# Task Execution Instructions

## Context Loading
1. Read PROGRESS.txt if it exists to understand what has been accomplished
2. Run `bd ready --json` to see available tasks ordered by priority

## Task Selection
Pick ONE task from the ready queue. Prefer:
- `verify-1` label
- Higher priority (lower number = higher priority)
- Tasks with fewer blockers
- Tasks that build on recently completed work noted in PROGRESS.txt

## Execution Protocol
For the selected task (e.g., bd-X):

1. **Claim**: `bd update bd-X in_progress`

2. **Implement**: Complete the task fully
   - Read relevant files
   - Make necessary changes
   - Follow existing code patterns

3. **Verify**: Run appropriate tests
   - `npm test` or equivalent
   - Manual verification if needed

4. **Commit**: 
```bash
   git add -A
   git commit -m "bd-X: <brief description>"
```

5. **Close**: `bd close bd-X "<completion summary>"`

6. **Document**: Append to PROGRESS.txt:
```
   ## [timestamp] Completed bd-X
   - What was done
   - Key decisions made
   - Suggestions for next tasks
```

## Stop Conditions
- Complete exactly ONE task per iteration
- If task cannot be completed, update it with blockers and move on
- If tests fail, debug and fix within this iteration

## Important
- Do not ask for clarification—make reasonable decisions
- Prefer small, atomic changes over large refactors
- Always verify before closing
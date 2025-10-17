# Shots on Goal - Data Model

## Overview

This document describes the data model for the shots-on-goal autonomous code migration system. The model is designed to be **append-only** and **event-based**, capturing the complete lifecycle of goals, attempts, tool calls, and validation.

## Design Principles

1. **Append-only**: Never UPDATE records after creation; only INSERT new records. This preserves complete history.
2. **Event-based**: Capture state transitions as events (attempt starts, attempt ends, validation runs).
3. **Normalized**: Use foreign keys and proper relationships rather than JSON blobs where possible.
4. **Explicit git state**: Track branches, worktrees, and merges as first-class entities.

## Core Tables

All tables have `id` (primary key) and `timestamp` (creation time) fields.

### session
```
session: id, timestamp, initial_goal, model_a, model_b, flags, repo_path, base_branch
```

Represents a single run of the system. Stores the initial parameters and configuration.

- `initial_goal`: The root goal description provided by the user
- `model_a`, `model_b`: The two models used for ping-pong collaboration
- `flags`: JSON containing CLI flags (max_tools, max_decompositions, etc.)
- `repo_path`: Absolute path to the repository
- `base_branch`: The branch to start from (detected automatically)

### tool
```
tool: id, timestamp, session_id, name, description
```

Session-scoped tool definitions. Each session gets fresh tool entries, since tool implementations can vary between sessions.

**Design decision**: Tools are NOT marked as read-only in the schema. The orchestration code decides which tools to make available to each attempt based on `attempt_type`.

### goal
```
goal: id, timestamp, session_id, goal_text, parent_goal_id, order, source, created_by_attempt_id
```

Goals represent work to be done. Goals can be:
- **Root goals**: Created from CLI input (`parent_goal_id` is NULL, `source='cli'`)
- **Sub-goals**: Created by breaking down a parent goal (`parent_goal_id` set, `created_by_attempt_id` points to the breakdown attempt)

- `order`: Sequence number among siblings (for sub-goals)
- `source`: How the goal was created (`cli` | `breakdown`)
- `created_by_attempt_id`: If created by breakdown, links to the attempt that created it

### validation_step
```
validation_step: id, timestamp, goal_id, order, command, source
```

Commands to validate whether a goal has been achieved. Expected to exit 0 on success.

- `order`: Sequence number (validation steps run in order)
- `command`: Shell command to execute (e.g., `bazel build //...`)
- `source`: How the step was defined (`cli` | `breakdown`)

### branch
```
branch: id, timestamp, session_id, name, parent_branch_id, parent_commit_sha, reason, created_by_goal_id
```

Git branches created during the session. Tracks the branching hierarchy.

- `parent_branch_id`: The branch this was created from
- `parent_commit_sha`: The commit SHA where this branch diverged
- `reason`: Why this branch was created (e.g., `goal`, `attempt`)
- `created_by_goal_id`: Which goal triggered this branch creation

### worktree
```
worktree: id, timestamp, branch_id, path, start_sha, reason
```

Git worktrees created for isolation. Each attempt gets its own worktree.

- `branch_id`: Which branch this worktree is tracking
- `path`: Filesystem path to the worktree
- `start_sha`: Initial commit SHA when worktree was created
- `reason`: Why this worktree was created

## Attempt Tables

### attempt
```
attempt: id, timestamp, goal_id, worktree_id, start_commit_sha, prompt, model, attempt_type
```

Represents an LLM invocation to work on a goal. Captures the **start state**.

- `goal_id`: Which goal this attempt is trying to achieve
- `worktree_id`: Which worktree the attempt runs in (derive branch via `worktree.branch_id`)
- `start_commit_sha`: Git commit SHA at the start of the attempt
- `prompt`: The prompt sent to the LLM
- `model`: Which model was used (e.g., `openrouter/anthropic/claude-sonnet-4.5`)
- `attempt_type`: Type of attempt:
  - `implementation`: Try to achieve the goal by making code changes
  - `breakdown`: Break down a goal into sub-goals (read-only tools)

**Design decision**: Breakdown is just another attempt type. This provides:
- Uniform timeout/error/tool-limit handling
- Same tool_call tracking
- Clear audit trail

**Design decision**: We removed `branch_id` from attempt since it can be derived via `worktree.branch_id`. This avoids data duplication and consistency issues.

### attempt_tool
```
attempt_tool: id, timestamp, attempt_id, tool_id
```

Junction table tracking which tools were made available to an attempt.

- For `implementation` attempts: read-write tools (list_directory, read_file, write_file, etc.)
- For `breakdown` attempts: read-only tools (list_directory, read_file, bazel_query, etc.)

### attempt_result
```
attempt_result: id, timestamp, attempt_id, end_commit_sha, diff, status, status_detail
```

Captures the **end state** of an attempt. Created after the LLM finishes.

- `end_commit_sha`: Git commit SHA at the end of the attempt
- `diff`: The git diff produced by this attempt (or pointer to stored diff)
- `status`: Outcome of the attempt:
  - `success`: Goal achieved, validation passed
  - `error`: Unexpected error occurred
  - `timeout`: LLM timed out (no activity for 2 minutes)
  - `tool_limit`: Tool use limit exceeded
  - `completed`: LLM finished but validation failed
- `status_detail`: Additional error message or details

### tool_call
```
tool_call: id, timestamp, attempt_id, order, tool_id, tool_name, input, output
```

Records each individual tool call made during an attempt.

- `order`: Sequence number (1st call, 2nd call, etc.)
- `tool_id`: Links to the tool definition (referential integrity)
- `tool_name`: Denormalized for convenience (logging/display)
- `input`: JSON of the tool's input arguments
- `output`: The tool's output/result

**Design decision**: We include both `tool_id` (for referential integrity) and `tool_name` (for convenience). This makes it easy to answer questions like "Was this call made with a different tool implementation?" while still being easy to log/display.

## Validation Tables

### validation_run
```
validation_run: id, timestamp, validation_step_id, attempt_result_id, exit_code, output
```

Records a single execution of a validation step.

- `validation_step_id`: Which validation step was run
- `attempt_result_id`: Which attempt result triggered this validation
- `exit_code`: The exit code of the validation command (0 = success)
- `output`: Captured stdout/stderr from the command

## Git Operation Tables

### merge
```
merge: id, timestamp, from_branch_id, from_commit_sha, to_branch_id, to_commit_sha, result_commit_sha
```

Records successful git merges. Only created when merge succeeds.

- `from_branch_id`: Source branch
- `from_commit_sha`: Commit from source branch
- `to_branch_id`: Target branch
- `to_commit_sha`: Commit on target branch before merge
- `result_commit_sha`: The merge commit created

**Design decision**: We only record successful merges. Failed merges are handled as errors in the orchestration code.

## Key Relationships

### Session → Goals → Attempts
```
session (1) ──→ (many) goals
goal (1) ──→ (many) attempts
attempt (1) ──→ (1) attempt_result
```

### Goal Hierarchy
```
goal (parent) ──→ (many) goals (children via parent_goal_id)
goal (created by) ←── attempt (via created_by_attempt_id)
```

### Git State
```
session (1) ──→ (many) branches
branch (1) ──→ (many) worktrees
attempt ──→ worktree ──→ branch
```

### Tools
```
session (1) ──→ (many) tools
attempt ←──→ tools (via attempt_tool junction table)
tool_call ──→ tool (via tool_id)
```

### Validation
```
goal (1) ──→ (many) validation_steps
validation_step (1) ──→ (many) validation_runs
validation_run ──→ attempt_result
```

## Typical Flow

### Implementation Attempt
1. Create `goal` (from CLI or breakdown)
2. Create `branch` for the goal
3. Create `worktree` for the attempt
4. Create `attempt` (type=`implementation`) with allowed tools
5. Create `attempt_tool` entries (read-write tools)
6. As LLM runs, create `tool_call` entries
7. Create `attempt_result` when LLM finishes
8. Create `validation_run` entries for each validation step
9. If successful, create `merge` to merge branch

### Goal Breakdown Flow
1. Implementation attempt fails
2. Create new `attempt` (type=`breakdown`) with read-only tools
3. Create `attempt_tool` entries (read-only tools only)
4. LLM analyzes and proposes sub-goals
5. Create `attempt_result` for breakdown attempt
6. Create new `goal` entries with:
   - `parent_goal_id` = failed goal
   - `created_by_attempt_id` = breakdown attempt
   - `source` = `breakdown`
7. Recursively work on sub-goals

## Query Examples

**Find all sub-goals created by a breakdown attempt:**
```sql
SELECT * FROM goal
WHERE created_by_attempt_id = ?
```

**Get the complete tool call history for an attempt:**
```sql
SELECT tc.*, t.description
FROM tool_call tc
JOIN tool t ON tc.tool_id = t.id
WHERE tc.attempt_id = ?
ORDER BY tc.order
```

**Find which tools were available but not used in an attempt:**
```sql
SELECT t.* FROM tool t
JOIN attempt_tool at ON t.id = at.tool_id
WHERE at.attempt_id = ?
  AND t.id NOT IN (
    SELECT tool_id FROM tool_call WHERE attempt_id = ?
  )
```

**Get the branch for an attempt:**
```sql
SELECT b.* FROM branch b
JOIN worktree w ON b.id = w.branch_id
JOIN attempt a ON w.id = a.worktree_id
WHERE a.id = ?
```

**Trace a goal's ancestry back to the root:**
```sql
WITH RECURSIVE goal_ancestry AS (
  SELECT id, parent_goal_id, goal_text, 0 as depth
  FROM goal WHERE id = ?
  UNION ALL
  SELECT g.id, g.parent_goal_id, g.goal_text, ga.depth + 1
  FROM goal g
  JOIN goal_ancestry ga ON g.id = ga.parent_goal_id
)
SELECT * FROM goal_ancestry ORDER BY depth DESC
```

## Design Decisions Summary

1. **Append-only architecture**: Enables complete audit trails and time-travel debugging
2. **Breakdown as attempt type**: Uniform handling, no special cases
3. **Normalized tools**: Session-scoped definitions + attempt-scoped availability
4. **No read-only flag on tools**: Orchestration code decides which tools to offer
5. **Tool ID + name in tool_call**: Referential integrity + logging convenience
6. **Branch derived from worktree in attempt**: Avoids duplication, single source of truth
7. **Validation is exit code based**: Simple, Unix-y, flexible
8. **Only record successful merges**: Failed merges are orchestration errors

## Future Considerations

- **Session resumability**: Add `session.end_time` and `session.final_status` to support resuming interrupted sessions
- **Worktree cleanup tracking**: Add `worktree.end_time` to track when worktrees are cleaned up
- **Decomposition metadata**: Consider adding a `breakdown_response` field to `attempt_result` to store the raw LLM response for breakdown attempts
- **Tool versioning**: If tool implementations need versioning within a session, add `tool.version`

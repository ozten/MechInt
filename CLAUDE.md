# Claude Code Notes

## Planning Approach

This project uses `bd` (BEADS) for planning and task management.

## What is BEADS?

BEADS is a memory upgrade for coding agents - a tiny, repo-local, git-backed database represented as JSONL with a CLI called `bd`. It solves LLM session amnesia, multi-agent coordination, and perma-lost TODOs discovered during execution.

## Key Features

- Issues stored as JSONL in `.beads/` directory
- Versioned, branched, and merged like code
- JSON output, dependency tracking, and auto-ready task detection
- Hash-based IDs (bd-a1b2) to prevent merge collisions in multi-agent/multi-branch workflows

## Installation

For macOS/Linux:
```bash
curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash
bd init
```

## Project Setup

This project has a Python virtual environment (`.venv/`) with the BEADS MCP server installed.

### At the start of each session:

Claude should activate the virtual environment (though currently using CLI instead of MCP):
```bash
source .venv/bin/activate
```

The BEADS MCP server is installed at `.venv/bin/python -m beads_mcp` but is not yet configured in Claude Code's MCP settings.

## Claude Code Integration

### Current Setup (CLI)
BEADS is currently accessed via the `bd` CLI tool. Claude uses Bash commands to interact with BEADS:

```bash
bd init                    # Initialize BEADS in current repo
bd create "Task title"     # Create a new task
bd list                    # List all tasks
bd list --status open      # List open tasks
bd ready                   # Show tasks ready to work on
bd show <id>              # Show task details
bd update <id>            # Update task
bd close <id>             # Close task
```

### Future Setup (MCP - Not Yet Configured)
When configured, BEADS will work with Claude Code through:
- MCP tools for structured task management
- Slash commands: `/bd-ready`, `/bd-create`, `/bd-show`, `/bd-update`, `/bd-close`

To enable MCP integration, add to Claude Code's MCP configuration:
```json
{
  "mcpServers": {
    "beads": {
      "command": "/Users/ozten/Projects/MechInt/.venv/bin/python",
      "args": ["-m", "beads_mcp"]
    }
  }
}
```

## Current Project: Grokking Phenomenon in Rust

This project implements the "grokking" experiment from Power et al. (2022) using Rust and wgpu for GPU acceleration.

### Project Goal
Train a small transformer to learn modular addition `(a + b) mod 97`, demonstrating the grokking phenomenon where:
1. Training accuracy hits 100% quickly (memorization)
2. Validation accuracy stays near random for thousands of steps
3. Validation accuracy suddenly jumps to 90%+ (generalization/"grokking")

### Key Implementation Details
- **Language**: Rust with wgpu for GPU compute shaders
- **Model**: 2-layer transformer (4 heads, dim=128, MLP hidden=512)
- **Training**: AdamW with high weight decay (1.0), 100k+ steps
- **Data**: All 9409 pairs of (a,b) with 50/50 train/val split

### Task Management
All implementation tasks are tracked in BEADS. Use `bd list --status open` to see current tasks, or `bd ready` to see tasks ready to work on.

## Why Use BEADS?

- Persistent memory across coding sessions
- Git-based versioning and merging
- Multi-agent coordination without conflicts
- Automatic task dependency tracking
- Works seamlessly with modern coding agents

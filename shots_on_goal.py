#!/usr/bin/env python3
"""
Shots on Goal - Autonomous goal-driven code migration system

Takes a goal and a git repository, recursively breaks down the goal,
makes changes, and validates them.
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import json
import logging
import time
import llm
from llm.utils import extract_fenced_code_block


# ============================================================================
# Git Utilities
# ============================================================================

def detect_default_branch(repo_path):
    """
    Detect the default branch of a git repository.

    Tries multiple methods:
    1. Check current branch
    2. Check remote HEAD (origin/HEAD)
    3. Fall back to common names (main, master)

    Args:
        repo_path: Path to the git repository

    Returns:
        The name of the default branch (e.g., 'main', 'master')

    Raises:
        ValueError: If no suitable branch can be detected
    """
    # Try method 1: Get the current branch
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        branch = result.stdout.strip()
        if branch and branch != 'HEAD':
            logging.debug(f"Detected current branch: {branch}")
            return branch
    except subprocess.CalledProcessError:
        pass

    # Try method 2: Check remote HEAD (origin/HEAD -> origin/main)
    try:
        result = subprocess.run(
            ['git', 'symbolic-ref', 'refs/remotes/origin/HEAD'],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        # Output is like "refs/remotes/origin/main"
        remote_head = result.stdout.strip()
        if remote_head.startswith('refs/remotes/origin/'):
            branch = remote_head.replace('refs/remotes/origin/', '')
            logging.debug(f"Detected default branch from origin/HEAD: {branch}")
            return branch
    except subprocess.CalledProcessError:
        pass

    # Try method 3: Check for common branch names
    try:
        result = subprocess.run(
            ['git', 'branch', '--list'],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        branches = [line.strip().lstrip('* ') for line in result.stdout.split('\n') if line.strip()]

        # Prefer main, then master, then take the first one
        for preferred in ['main', 'master']:
            if preferred in branches:
                logging.debug(f"Detected branch from common names: {preferred}")
                return preferred

        # If we have any branches, use the first
        if branches:
            logging.debug(f"Using first available branch: {branches[0]}")
            return branches[0]
    except subprocess.CalledProcessError:
        pass

    # If all methods fail, raise an error
    raise ValueError(f"Could not detect default branch in repository: {repo_path}")


# ============================================================================
# Timeout Utilities
# ============================================================================

class LLMTimeoutError(Exception):
    """Raised when an LLM call exceeds the timeout."""
    pass



# ============================================================================
# Database Setup V2 (New Schema)
# ============================================================================

def init_database_v2(db_path):
    """
    Initialize SQLite database with V2 schema.
    This is the new append-only, event-based schema.
    Returns connection object.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Enable foreign key constraints
    db.execute("PRAGMA foreign_keys = ON")

    cursor = db.cursor()

    # Session table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            initial_goal TEXT NOT NULL,
            model_a TEXT NOT NULL,
            model_b TEXT NOT NULL,
            flags TEXT,  -- JSON
            repo_path TEXT NOT NULL,
            base_branch TEXT NOT NULL
        )
    """)

    # Tool table (session-scoped tool definitions)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (session_id) REFERENCES session(id)
        )
    """)

    # Goal table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS goal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id INTEGER NOT NULL,
            goal_text TEXT NOT NULL,
            parent_goal_id INTEGER,
            order_num INTEGER,  -- renamed from 'order' to avoid SQL keyword
            source TEXT,  -- 'cli' or 'breakdown'
            created_by_attempt_id INTEGER,
            FOREIGN KEY (session_id) REFERENCES session(id),
            FOREIGN KEY (parent_goal_id) REFERENCES goal(id),
            FOREIGN KEY (created_by_attempt_id) REFERENCES attempt(id)
        )
    """)

    # Validation step table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_step (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            goal_id INTEGER NOT NULL,
            order_num INTEGER NOT NULL,
            command TEXT NOT NULL,
            source TEXT,
            FOREIGN KEY (goal_id) REFERENCES goal(id)
        )
    """)

    # Branch table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS branch (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            parent_branch_id INTEGER,
            parent_commit_sha TEXT,
            reason TEXT,
            created_by_goal_id INTEGER,
            FOREIGN KEY (session_id) REFERENCES session(id),
            FOREIGN KEY (parent_branch_id) REFERENCES branch(id),
            FOREIGN KEY (created_by_goal_id) REFERENCES goal(id)
        )
    """)

    # Worktree table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS worktree (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            branch_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            start_sha TEXT NOT NULL,
            reason TEXT,
            FOREIGN KEY (branch_id) REFERENCES branch(id)
        )
    """)

    # Attempt table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            goal_id INTEGER NOT NULL,
            worktree_id INTEGER NOT NULL,
            start_commit_sha TEXT NOT NULL,
            prompt TEXT NOT NULL,
            model TEXT NOT NULL,
            attempt_type TEXT NOT NULL,  -- 'implementation' or 'breakdown'
            FOREIGN KEY (goal_id) REFERENCES goal(id),
            FOREIGN KEY (worktree_id) REFERENCES worktree(id)
        )
    """)

    # Attempt tool junction table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempt_tool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attempt_id INTEGER NOT NULL,
            tool_id INTEGER NOT NULL,
            FOREIGN KEY (attempt_id) REFERENCES attempt(id),
            FOREIGN KEY (tool_id) REFERENCES tool(id)
        )
    """)

    # Attempt result table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempt_result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attempt_id INTEGER NOT NULL,
            end_commit_sha TEXT,
            diff TEXT,
            status TEXT NOT NULL,  -- 'success', 'error', 'timeout', 'tool_limit', 'completed'
            status_detail TEXT,
            FOREIGN KEY (attempt_id) REFERENCES attempt(id)
        )
    """)

    # Tool call table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_call (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attempt_id INTEGER NOT NULL,
            order_num INTEGER NOT NULL,
            tool_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL,  -- denormalized for convenience
            input TEXT,  -- JSON
            output TEXT,
            FOREIGN KEY (attempt_id) REFERENCES attempt(id),
            FOREIGN KEY (tool_id) REFERENCES tool(id)
        )
    """)

    # Validation run table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            validation_step_id INTEGER NOT NULL,
            attempt_result_id INTEGER NOT NULL,
            exit_code INTEGER NOT NULL,
            output TEXT,
            FOREIGN KEY (validation_step_id) REFERENCES validation_step(id),
            FOREIGN KEY (attempt_result_id) REFERENCES attempt_result(id)
        )
    """)

    # Merge table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS merge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            from_branch_id INTEGER NOT NULL,
            from_commit_sha TEXT NOT NULL,
            to_branch_id INTEGER NOT NULL,
            to_commit_sha TEXT NOT NULL,
            result_commit_sha TEXT NOT NULL,
            FOREIGN KEY (from_branch_id) REFERENCES branch(id),
            FOREIGN KEY (to_branch_id) REFERENCES branch(id)
        )
    """)

    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_goal_session ON goal(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_goal_parent ON goal(parent_goal_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempt_goal ON attempt(goal_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_call_attempt ON tool_call(attempt_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_validation_run_step ON validation_run(validation_step_id)")

    db.commit()
    return db


# ============================================================================
# Database Helper Functions V2
# ============================================================================

def create_session_v2(db, initial_goal, model_a, model_b, flags, repo_path, base_branch):
    """Create a session record in V2 schema."""
    cursor = db.execute(
        """
        INSERT INTO session (initial_goal, model_a, model_b, flags, repo_path, base_branch)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (initial_goal, model_a, model_b, json.dumps(flags), repo_path, base_branch)
    )
    db.commit()
    return cursor.lastrowid


def get_session_v2(db, session_id):
    """Get a session by ID."""
    row = db.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def create_tool_v2(db, session_id, name, description=""):
    """Create a tool definition."""
    cursor = db.execute(
        """
        INSERT INTO tool (session_id, name, description)
        VALUES (?, ?, ?)
        """,
        (session_id, name, description)
    )
    db.commit()
    return cursor.lastrowid


def get_tool_v2(db, tool_id):
    """Get a tool by ID."""
    row = db.execute("SELECT * FROM tool WHERE id = ?", (tool_id,)).fetchone()
    return dict(row) if row else None


def get_tools_for_session_v2(db, session_id):
    """Get all tools for a session."""
    rows = db.execute("SELECT * FROM tool WHERE session_id = ?", (session_id,)).fetchall()
    return [dict(row) for row in rows]


def create_goal_v2(db, session_id, goal_text, parent_goal_id=None, order_num=None,
                   source='cli', created_by_attempt_id=None):
    """Create a goal record."""
    cursor = db.execute(
        """
        INSERT INTO goal (session_id, goal_text, parent_goal_id, order_num, source, created_by_attempt_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, goal_text, parent_goal_id, order_num, source, created_by_attempt_id)
    )
    db.commit()
    return cursor.lastrowid


def get_goal_v2(db, goal_id):
    """Get a goal by ID."""
    row = db.execute("SELECT * FROM goal WHERE id = ?", (goal_id,)).fetchone()
    return dict(row) if row else None


def get_child_goals_v2(db, parent_goal_id):
    """Get all child goals of a parent goal, ordered by order_num."""
    rows = db.execute(
        "SELECT * FROM goal WHERE parent_goal_id = ? ORDER BY order_num",
        (parent_goal_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def create_validation_step_v2(db, goal_id, order_num, command, source='cli'):
    """Create a validation step."""
    cursor = db.execute(
        """
        INSERT INTO validation_step (goal_id, order_num, command, source)
        VALUES (?, ?, ?, ?)
        """,
        (goal_id, order_num, command, source)
    )
    db.commit()
    return cursor.lastrowid


def get_validation_steps_v2(db, goal_id):
    """Get all validation steps for a goal, ordered by order_num."""
    rows = db.execute(
        "SELECT * FROM validation_step WHERE goal_id = ? ORDER BY order_num",
        (goal_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def create_branch_v2(db, session_id, name, parent_branch_id=None,
                     parent_commit_sha=None, reason=None, created_by_goal_id=None):
    """Create a branch record."""
    cursor = db.execute(
        """
        INSERT INTO branch (session_id, name, parent_branch_id, parent_commit_sha, reason, created_by_goal_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, name, parent_branch_id, parent_commit_sha, reason, created_by_goal_id)
    )
    db.commit()
    return cursor.lastrowid


def get_branch_v2(db, branch_id):
    """Get a branch by ID."""
    row = db.execute("SELECT * FROM branch WHERE id = ?", (branch_id,)).fetchone()
    return dict(row) if row else None


def get_branch_by_name_v2(db, session_id, name):
    """Get a branch by name within a session."""
    row = db.execute(
        "SELECT * FROM branch WHERE session_id = ? AND name = ?",
        (session_id, name)
    ).fetchone()
    return dict(row) if row else None


def create_worktree_v2(db, branch_id, path, start_sha, reason=None):
    """Create a worktree record."""
    cursor = db.execute(
        """
        INSERT INTO worktree (branch_id, path, start_sha, reason)
        VALUES (?, ?, ?, ?)
        """,
        (branch_id, path, start_sha, reason)
    )
    db.commit()
    return cursor.lastrowid


def get_worktree_v2(db, worktree_id):
    """Get a worktree by ID."""
    row = db.execute("SELECT * FROM worktree WHERE id = ?", (worktree_id,)).fetchone()
    return dict(row) if row else None


def create_attempt_v2(db, goal_id, worktree_id, start_commit_sha, prompt, model, attempt_type):
    """Create an attempt record."""
    cursor = db.execute(
        """
        INSERT INTO attempt (goal_id, worktree_id, start_commit_sha, prompt, model, attempt_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (goal_id, worktree_id, start_commit_sha, prompt, model, attempt_type)
    )
    db.commit()
    return cursor.lastrowid


def get_attempt_v2(db, attempt_id):
    """Get an attempt by ID."""
    row = db.execute("SELECT * FROM attempt WHERE id = ?", (attempt_id,)).fetchone()
    return dict(row) if row else None


def get_attempts_for_goal_v2(db, goal_id):
    """Get all attempts for a goal."""
    rows = db.execute(
        "SELECT * FROM attempt WHERE goal_id = ? ORDER BY timestamp",
        (goal_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def create_attempt_tool_v2(db, attempt_id, tool_id):
    """Link a tool to an attempt."""
    cursor = db.execute(
        """
        INSERT INTO attempt_tool (attempt_id, tool_id)
        VALUES (?, ?)
        """,
        (attempt_id, tool_id)
    )
    db.commit()
    return cursor.lastrowid


def get_tools_for_attempt_v2(db, attempt_id):
    """Get all tools available to an attempt."""
    rows = db.execute(
        """
        SELECT t.* FROM tool t
        JOIN attempt_tool at ON t.id = at.tool_id
        WHERE at.attempt_id = ?
        ORDER BY t.name
        """,
        (attempt_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def create_attempt_result_v2(db, attempt_id, end_commit_sha, diff, status, status_detail=None):
    """Create an attempt result record."""
    cursor = db.execute(
        """
        INSERT INTO attempt_result (attempt_id, end_commit_sha, diff, status, status_detail)
        VALUES (?, ?, ?, ?, ?)
        """,
        (attempt_id, end_commit_sha, diff, status, status_detail)
    )
    db.commit()
    return cursor.lastrowid


def get_attempt_result_v2(db, attempt_result_id):
    """Get an attempt result by ID."""
    row = db.execute("SELECT * FROM attempt_result WHERE id = ?", (attempt_result_id,)).fetchone()
    return dict(row) if row else None


def get_attempt_result_by_attempt_v2(db, attempt_id):
    """Get the attempt result for an attempt."""
    row = db.execute(
        "SELECT * FROM attempt_result WHERE attempt_id = ?",
        (attempt_id,)
    ).fetchone()
    return dict(row) if row else None


def create_tool_call_v2(db, attempt_id, order_num, tool_id, tool_name, input_data, output):
    """Create a tool call record."""
    cursor = db.execute(
        """
        INSERT INTO tool_call (attempt_id, order_num, tool_id, tool_name, input, output)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (attempt_id, order_num, tool_id, tool_name, input_data, output)
    )
    db.commit()
    return cursor.lastrowid


def get_tool_calls_v2(db, attempt_id):
    """Get all tool calls for an attempt, ordered by order_num."""
    rows = db.execute(
        "SELECT * FROM tool_call WHERE attempt_id = ? ORDER BY order_num",
        (attempt_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def create_validation_run_v2(db, validation_step_id, attempt_result_id, exit_code, output):
    """Create a validation run record."""
    cursor = db.execute(
        """
        INSERT INTO validation_run (validation_step_id, attempt_result_id, exit_code, output)
        VALUES (?, ?, ?, ?)
        """,
        (validation_step_id, attempt_result_id, exit_code, output)
    )
    db.commit()
    return cursor.lastrowid


def get_validation_runs_v2(db, attempt_result_id):
    """Get all validation runs for an attempt result."""
    rows = db.execute(
        "SELECT * FROM validation_run WHERE attempt_result_id = ? ORDER BY timestamp",
        (attempt_result_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def create_merge_v2(db, from_branch_id, from_commit_sha, to_branch_id, to_commit_sha, result_commit_sha):
    """Create a merge record."""
    cursor = db.execute(
        """
        INSERT INTO merge (from_branch_id, from_commit_sha, to_branch_id, to_commit_sha, result_commit_sha)
        VALUES (?, ?, ?, ?, ?)
        """,
        (from_branch_id, from_commit_sha, to_branch_id, to_commit_sha, result_commit_sha)
    )
    db.commit()
    return cursor.lastrowid


def get_merge_v2(db, merge_id):
    """Get a merge by ID."""
    row = db.execute("SELECT * FROM merge WHERE id = ?", (merge_id,)).fetchone()
    return dict(row) if row else None


# ============================================================================
# Query Helpers V2 (Higher-level queries)
# ============================================================================

def get_goal_with_attempts_v2(db, goal_id):
    """
    Get a goal with all its attempts and their results.

    Returns:
        {
            'goal': {...},
            'attempts': [
                {
                    'attempt': {...},
                    'result': {...} or None,
                    'tool_call_count': int
                },
                ...
            ]
        }
    """
    goal = get_goal_v2(db, goal_id)
    if not goal:
        return None

    attempts = []
    for attempt in get_attempts_for_goal_v2(db, goal_id):
        result = get_attempt_result_by_attempt_v2(db, attempt['id'])
        tool_calls = get_tool_calls_v2(db, attempt['id'])

        attempts.append({
            'attempt': attempt,
            'result': result,
            'tool_call_count': len(tool_calls)
        })

    return {
        'goal': goal,
        'attempts': attempts
    }


def get_attempt_with_details_v2(db, attempt_id):
    """
    Get an attempt with all related information.

    Returns:
        {
            'attempt': {...},
            'goal': {...},
            'worktree': {...},
            'branch': {...},
            'tools': [{...}, ...],
            'tool_calls': [{...}, ...],
            'result': {...} or None,
            'validation_runs': [{...}, ...] if result exists
        }
    """
    attempt = get_attempt_v2(db, attempt_id)
    if not attempt:
        return None

    goal = get_goal_v2(db, attempt['goal_id'])
    worktree = get_worktree_v2(db, attempt['worktree_id'])
    branch = get_branch_v2(db, worktree['branch_id']) if worktree else None
    tools = get_tools_for_attempt_v2(db, attempt_id)
    tool_calls = get_tool_calls_v2(db, attempt_id)
    result = get_attempt_result_by_attempt_v2(db, attempt_id)

    validation_runs = []
    if result:
        validation_runs = get_validation_runs_v2(db, result['id'])

    return {
        'attempt': attempt,
        'goal': goal,
        'worktree': worktree,
        'branch': branch,
        'tools': tools,
        'tool_calls': tool_calls,
        'result': result,
        'validation_runs': validation_runs
    }


def get_goal_tree_v2(db, goal_id):
    """
    Get a goal and all its descendants recursively.

    Returns:
        {
            'goal': {...},
            'children': [
                {
                    'goal': {...},
                    'children': [...]
                },
                ...
            ]
        }
    """
    goal = get_goal_v2(db, goal_id)
    if not goal:
        return None

    children = []
    for child_goal in get_child_goals_v2(db, goal_id):
        child_tree = get_goal_tree_v2(db, child_goal['id'])
        if child_tree:
            children.append(child_tree)

    return {
        'goal': goal,
        'children': children
    }


def get_validation_status_v2(db, goal_id):
    """
    Check validation status for a goal's latest attempt.

    Returns:
        {
            'has_validation': bool,
            'all_passed': bool or None,
            'step_count': int,
            'passed_count': int,
            'failed_count': int,
            'steps': [
                {
                    'step': {...},
                    'run': {...} or None,
                    'passed': bool or None
                },
                ...
            ]
        }
    """
    # Get validation steps for this goal
    validation_steps = get_validation_steps_v2(db, goal_id)

    if not validation_steps:
        return {
            'has_validation': False,
            'all_passed': None,
            'step_count': 0,
            'passed_count': 0,
            'failed_count': 0,
            'steps': []
        }

    # Get the latest attempt for this goal
    attempts = get_attempts_for_goal_v2(db, goal_id)
    if not attempts:
        return {
            'has_validation': True,
            'all_passed': None,
            'step_count': len(validation_steps),
            'passed_count': 0,
            'failed_count': 0,
            'steps': [{'step': step, 'run': None, 'passed': None} for step in validation_steps]
        }

    latest_attempt = attempts[-1]  # Last one (ordered by timestamp)
    result = get_attempt_result_by_attempt_v2(db, latest_attempt['id'])

    if not result:
        return {
            'has_validation': True,
            'all_passed': None,
            'step_count': len(validation_steps),
            'passed_count': 0,
            'failed_count': 0,
            'steps': [{'step': step, 'run': None, 'passed': None} for step in validation_steps]
        }

    # Get validation runs for this result
    validation_runs = get_validation_runs_v2(db, result['id'])
    runs_by_step = {run['validation_step_id']: run for run in validation_runs}

    # Build status for each step
    steps_status = []
    passed_count = 0
    failed_count = 0

    for step in validation_steps:
        run = runs_by_step.get(step['id'])
        passed = run['exit_code'] == 0 if run else None

        if passed is True:
            passed_count += 1
        elif passed is False:
            failed_count += 1

        steps_status.append({
            'step': step,
            'run': run,
            'passed': passed
        })

    all_passed = passed_count == len(validation_steps) if validation_runs else None

    return {
        'has_validation': True,
        'all_passed': all_passed,
        'step_count': len(validation_steps),
        'passed_count': passed_count,
        'failed_count': failed_count,
        'steps': steps_status
    }


def get_goal_ancestry_v2(db, goal_id):
    """
    Get the ancestry chain from goal back to root.

    Returns:
        [root_goal, ..., parent_goal, goal]
    """
    ancestry = []
    current_id = goal_id

    while current_id:
        goal = get_goal_v2(db, current_id)
        if not goal:
            break
        ancestry.insert(0, goal)  # Prepend to build root-to-leaf order
        current_id = goal['parent_goal_id']

    return ancestry


def get_attempts_summary_v2(db, goal_id):
    """
    Get summary statistics for all attempts on a goal.

    Returns:
        {
            'total_attempts': int,
            'by_model': {model_name: count, ...},
            'by_type': {attempt_type: count, ...},
            'by_status': {status: count, ...},
            'total_tool_calls': int
        }
    """
    attempts = get_attempts_for_goal_v2(db, goal_id)

    by_model = {}
    by_type = {}
    by_status = {}
    total_tool_calls = 0

    for attempt in attempts:
        # Count by model
        model = attempt['model']
        by_model[model] = by_model.get(model, 0) + 1

        # Count by type
        attempt_type = attempt['attempt_type']
        by_type[attempt_type] = by_type.get(attempt_type, 0) + 1

        # Count tool calls
        tool_calls = get_tool_calls_v2(db, attempt['id'])
        total_tool_calls += len(tool_calls)

        # Count by status (if result exists)
        result = get_attempt_result_by_attempt_v2(db, attempt['id'])
        if result:
            status = result['status']
            by_status[status] = by_status.get(status, 0) + 1

    return {
        'total_attempts': len(attempts),
        'by_model': by_model,
        'by_type': by_type,
        'by_status': by_status,
        'total_tool_calls': total_tool_calls
    }


# ============================================================================
# Orchestration V2 - Tool Execution
# ============================================================================

class WorktreeToolExecutor:
    """Executes tools directly in a worktree (no container isolation)."""

    def __init__(self, worktree_path):
        """
        Initialize with a worktree path.

        Args:
            worktree_path: Path to the git worktree
        """
        self.worktree_path = worktree_path

    def _run_command(self, cmd, timeout=60):
        """
        Run a shell command in the worktree.

        Args:
            cmd: Command string to execute
            timeout: Timeout in seconds

        Returns:
            dict with 'success', 'stdout', 'stderr', 'exit_code'
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=self.worktree_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout}s',
                'exit_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1
            }

    # File Operations

    def read_file(self, path):
        """Read a file from the worktree."""
        try:
            full_path = os.path.join(self.worktree_path, path)
            with open(full_path, 'r') as f:
                content = f.read()
            return {'success': True, 'content': content, 'error': None}
        except Exception as e:
            return {'success': False, 'content': None, 'error': str(e)}

    def read_multiple_files(self, paths):
        """
        Read multiple files from the worktree at once.

        Args:
            paths: List of file paths relative to worktree root

        Returns:
            dict mapping path to result dict with 'success', 'content', 'error'
        """
        results = {}
        for path in paths:
            try:
                full_path = os.path.join(self.worktree_path, path)
                with open(full_path, 'r') as f:
                    content = f.read()
                results[path] = {'success': True, 'content': content, 'error': None}
            except Exception as e:
                results[path] = {'success': False, 'content': None, 'error': str(e)}
        return results

    def write_file(self, path, content):
        """Write content to a file in the worktree."""
        try:
            full_path = os.path.join(self.worktree_path, path)
            # Create parent directories if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            return {'success': True, 'error': None}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def write_multiple_files(self, files_dict):
        """
        Write multiple files at once.

        Args:
            files_dict: Dict mapping file paths to content strings

        Returns:
            dict mapping path to result dict with 'success' and 'error'
        """
        results = {}
        for path, content in files_dict.items():
            results[path] = self.write_file(path, content)
        return results

    def find_replace_in_file(self, path, old_text, new_text):
        """Find and replace text in a file (must match exactly once)."""
        try:
            full_path = os.path.join(self.worktree_path, path)
            with open(full_path, 'r') as f:
                content = f.read()

            count = content.count(old_text)
            if count == 0:
                return {'success': False, 'error': f'Text not found in {path}'}
            elif count > 1:
                return {'success': False, 'error': f'Text appears {count} times (expected exactly 1)'}

            new_content = content.replace(old_text, new_text)
            with open(full_path, 'w') as f:
                f.write(new_content)

            return {'success': True, 'error': None}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def list_directory(self, path="."):
        """List files in a directory."""
        try:
            full_path = os.path.join(self.worktree_path, path)
            files = os.listdir(full_path)
            return {'success': True, 'files': sorted(files), 'error': None}
        except Exception as e:
            return {'success': False, 'files': [], 'error': str(e)}

    def find_files(self, pattern, path="."):
        """Find files by glob pattern."""
        result = self._run_command(f'find {path} -name "{pattern}" -type f', timeout=30)
        if result['success']:
            files = [f.strip() for f in result['stdout'].split('\n') if f.strip()]
            return {'success': True, 'files': files, 'error': None}
        else:
            return {'success': False, 'files': [], 'error': result['stderr']}

    def ripgrep(self, pattern, path=".", glob=None, ignore_case=False):
        """Search code using ripgrep."""
        cmd = f'rg --json "{pattern}" {path}'
        if glob:
            cmd += f' -g "{glob}"'
        if ignore_case:
            cmd += ' -i'

        result = self._run_command(cmd, timeout=30)
        return result

    # Bazel Tools

    def bazel_build(self, targets="//...", flags=None):
        """Build Bazel targets."""
        cmd = f"bazel build {targets}"
        if flags:
            cmd += " " + " ".join(flags)
        return self._run_command(cmd, timeout=600)

    def bazel_test(self, targets="//...", flags=None):
        """Run Bazel tests."""
        cmd = f"bazel test {targets}"
        if flags:
            cmd += " " + " ".join(flags)
        return self._run_command(cmd, timeout=600)

    def bazel_query(self, query):
        """Query the Bazel build graph."""
        query_escaped = query.replace('"', '\\"')
        cmd = f'bazel query "{query_escaped}"'
        return self._run_command(cmd, timeout=60)


def create_v2_tool_functions(tools_executor):
    """
    Create LLM tool functions for V2 (using WorktreeToolExecutor).

    Args:
        tools_executor: WorktreeToolExecutor instance

    Returns:
        List of tool functions with docstrings
    """

    def read_file(path: str) -> str:
        """
        Read a file from the workspace.

        Args:
            path: Path to file relative to workspace root

        Returns:
            File contents as a string
        """
        result = tools_executor.read_file(path)
        if result['success']:
            return result['content']
        else:
            return f"ERROR: {result['error']}"

    def read_multiple_files(paths: list) -> str:
        """
        Read multiple files at once (more efficient than calling read_file multiple times).

        Args:
            paths: List of file paths relative to workspace root

        Returns:
            Formatted string with contents of each file, separated by headers
        """
        results = tools_executor.read_multiple_files(paths)
        output = []
        for path, result in results.items():
            if result['success']:
                output.append(f"=== {path} ===\n{result['content']}")
            else:
                output.append(f"=== {path} ===\nERROR: {result['error']}")
        return "\n\n".join(output) if output else "No files read"

    def write_file(path: str, content: str) -> str:
        """
        Write content to a file in the workspace.

        Args:
            path: Path to file relative to workspace root
            content: Content to write to the file

        Returns:
            Success message or error
        """
        result = tools_executor.write_file(path, content)
        if result['success']:
            return f"Successfully wrote to {path}"
        else:
            return f"ERROR: {result['error']}"

    def write_multiple_files(files: dict) -> str:
        """
        Write multiple files at once (more efficient than calling write_file multiple times).

        Args:
            files: Dict mapping file paths to content strings
                   Example: {"src/foo.py": "content1", "src/bar.py": "content2"}

        Returns:
            Summary of files written or errors
        """
        results = tools_executor.write_multiple_files(files)
        output = []
        success_count = 0
        for path, result in results.items():
            if result['success']:
                success_count += 1
                output.append(f"✓ {path}")
            else:
                output.append(f"✗ {path}: {result['error']}")

        summary = f"Wrote {success_count}/{len(results)} files successfully"
        if output:
            return f"{summary}\n" + "\n".join(output)
        return summary

    def find_replace_in_file(path: str, old_text: str, new_text: str) -> str:
        """
        Find and replace text in a file. Requires exactly one match.

        Args:
            path: Path to file relative to workspace root
            old_text: Text to find (must match exactly once)
            new_text: Text to replace with

        Returns:
            Success message or error
        """
        result = tools_executor.find_replace_in_file(path, old_text, new_text)
        if result['success']:
            return f"Successfully replaced text in {path}"
        else:
            return f"ERROR: {result['error']}"

    def list_directory(path: str = ".") -> str:
        """
        List files in a directory.

        Args:
            path: Directory path (default: current directory)

        Returns:
            Newline-separated list of files
        """
        result = tools_executor.list_directory(path)
        if result['success']:
            return "\n".join(result['files'])
        else:
            return f"ERROR: {result['error']}"

    def find_files(pattern: str, path: str = ".") -> str:
        """
        Find files by name pattern.

        Args:
            pattern: Filename pattern (glob style, e.g., "*.py" or "BUILD*")
            path: Path to search in (default: current directory)

        Returns:
            Newline-separated list of matching file paths
        """
        result = tools_executor.find_files(pattern, path)
        if result['success']:
            return "\n".join(result['files']) if result['files'] else "No files found"
        else:
            return f"ERROR: {result['error']}"

    def ripgrep(pattern: str, path: str = ".", glob: str = None, ignore_case: bool = False) -> str:
        """
        Search code using ripgrep.

        Args:
            pattern: Search pattern (regex)
            path: Path to search in (default: current directory)
            glob: Optional glob pattern to filter files (e.g., "*.py")
            ignore_case: Case-insensitive search (default: false)

        Returns:
            Search results in JSON format
        """
        result = tools_executor.ripgrep(pattern, path, glob, ignore_case)
        if result['success']:
            return result['stdout'] if result['stdout'] else "No matches found"
        else:
            return f"ERROR: {result['stderr']}"

    def bazel_build(targets: str = "//...", flags: str = None) -> str:
        """
        Build Bazel targets.

        Args:
            targets: Bazel target pattern (default: //...)
            flags: Optional space-separated bazel flags

        Returns:
            Build output or error message
        """
        flag_list = flags.split() if flags else None
        result = tools_executor.bazel_build(targets, flag_list)

        output = []
        if result['stdout']:
            output.append(result['stdout'])
        if result['stderr']:
            output.append(result['stderr'])

        if result['success']:
            return "\n".join(output) or "Build succeeded"
        else:
            return f"Build failed (exit code {result['exit_code']}):\n" + "\n".join(output)

    def bazel_test(targets: str = "//...", flags: str = None) -> str:
        """
        Run Bazel tests.

        Args:
            targets: Bazel test target pattern (default: //...)
            flags: Optional space-separated bazel flags

        Returns:
            Test output or error message
        """
        flag_list = flags.split() if flags else None
        result = tools_executor.bazel_test(targets, flag_list)

        output = []
        if result['stdout']:
            output.append(result['stdout'])
        if result['stderr']:
            output.append(result['stderr'])

        if result['success']:
            return "\n".join(output) or "Tests passed"
        else:
            return f"Tests failed (exit code {result['exit_code']}):\n" + "\n".join(output)

    def bazel_query(query: str) -> str:
        """
        Query the Bazel build graph.

        Args:
            query: Bazel query expression (e.g., "//..." or "deps(//pkg:target)")

        Returns:
            Query results or error message
        """
        result = tools_executor.bazel_query(query)

        output = []
        if result['stdout']:
            output.append(result['stdout'])
        if result['stderr']:
            output.append(result['stderr'])

        if result['success']:
            return "\n".join(output) or "Query succeeded"
        else:
            return f"Query failed (exit code {result['exit_code']}):\n" + "\n".join(output)

    return [
        read_file,
        read_multiple_files,
        write_file,
        write_multiple_files,
        find_replace_in_file,
        list_directory,
        find_files,
        ripgrep,
        bazel_build,
        bazel_test,
        bazel_query,
    ]


# ============================================================================
# Orchestration V2 (with real LLM integration)
# ============================================================================

def work_on_goal_v2_simple(db, session_id, goal_id, repo_path, model_id,
                           attempt_type='implementation', max_tools=50,
                           image='shots-on-goal:latest', runtime='container'):
    """
    V2 orchestration with real LLM integration.

    This function:
    - Creates branch and worktree
    - Tracks attempt in V2 schema
    - Calls real LLM with tools (in container)
    - Records tool calls
    - Runs validation
    - Creates attempt result

    Args:
        db: Database connection
        session_id: Session ID
        goal_id: Goal to work on
        repo_path: Repository path
        model_id: Model to use
        attempt_type: 'implementation' or 'breakdown'
        max_tools: Max tool calls allowed
        image: Container image to use
        runtime: Container runtime ('container' or 'docker')

    Returns:
        {
            'success': bool,
            'attempt_id': int,
            'result_id': int,
            'validation_passed': bool or None
        }
    """
    import subprocess

    # Get goal and session
    goal = get_goal_v2(db, goal_id)
    session = get_session_v2(db, session_id)

    if not goal or not session:
        raise ValueError("Goal or session not found")

    logging.info(f"[V2] Working on goal {goal_id}: {goal['goal_text'][:60]}")

    # Get or create persistent goal branch: s{session}-g{goal}
    goal_branch_name = f"s{session_id}-g{goal_id}"

    # Check if goal branch exists in git
    check_branch = subprocess.run(
        ['git', 'rev-parse', '--verify', goal_branch_name],
        cwd=repo_path,
        capture_output=True,
        text=True
    )

    if check_branch.returncode == 0:
        # Goal branch exists - use its current HEAD
        logging.info(f"[V2] Using existing goal branch: {goal_branch_name}")
        goal_branch_sha = check_branch.stdout.strip()
    else:
        # Create new goal branch from session base branch
        base_branch = session['base_branch']
        result = subprocess.run(
            ['git', 'rev-parse', base_branch],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        base_sha = result.stdout.strip()

        subprocess.run(
            ['git', 'branch', goal_branch_name, base_sha],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        logging.info(f"[V2] Created new goal branch: {goal_branch_name} from {base_branch}")
        goal_branch_sha = base_sha

        # Track goal branch in database
        create_branch_v2(
            db,
            session_id=session_id,
            name=goal_branch_name,
            parent_commit_sha=base_sha,
            reason=f"Persistent branch for goal {goal_id}",
            created_by_goal_id=goal_id
        )

    # Get next attempt ID for unique paths
    cursor = db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM attempt")
    next_attempt_id = cursor.fetchone()[0]

    # Create unique attempt branch name (includes timestamp)
    timestamp = int(time.time())
    attempt_branch_name = f"s{session_id}-g{goal_id}-a{next_attempt_id}-{timestamp}"

    # Create attempt branch from goal branch
    attempt_branch_id = create_branch_v2(
        db,
        session_id=session_id,
        name=attempt_branch_name,
        parent_commit_sha=goal_branch_sha,
        reason=f"Attempt for goal {goal_id}",
        created_by_goal_id=goal_id
    )
    logging.info(f"[V2] Created attempt branch: {attempt_branch_name} from {goal_branch_name}")

    # Create worktree path (globally unique: session + goal + attempt + timestamp)
    worktree_path = f"{repo_path}/worktrees/s{session_id}-g{goal_id}-a{next_attempt_id}-{timestamp}"

    # Create git worktree with attempt branch
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)
    subprocess.run(
        ['git', 'worktree', 'add', '-b', attempt_branch_name, worktree_path, goal_branch_sha],
        cwd=repo_path,
        check=True,
        capture_output=True
    )
    logging.info(f"[V2] Created git worktree: {worktree_path} (branch: {attempt_branch_name})")

    # Configure git for commits (defensive - may already be set)
    subprocess.run(
        ['git', 'config', 'user.email', 'shots-on-goal@example.com'],
        cwd=worktree_path,
        capture_output=True
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Shots on Goal V2'],
        cwd=worktree_path,
        capture_output=True
    )

    # Track worktree in database
    worktree_id = create_worktree_v2(
        db,
        branch_id=attempt_branch_id,
        path=worktree_path,
        start_sha=goal_branch_sha,
        reason=f"Attempt for goal {goal_id}"
    )

    # Create attempt
    prompt = f"Goal: {goal['goal_text']}\n\nPlease complete this goal."
    attempt_id = create_attempt_v2(
        db,
        goal_id=goal_id,
        worktree_id=worktree_id,
        start_commit_sha=goal_branch_sha,
        prompt=prompt,
        model=model_id,
        attempt_type=attempt_type
    )
    logging.info(f"[V2] Created attempt {attempt_id} with {model_id}")

    # Get tools for this session and link to attempt
    tools_list = get_tools_for_session_v2(db, session_id)
    tool_id_map = {}  # Map tool names to IDs for recording
    for tool in tools_list:
        create_attempt_tool_v2(db, attempt_id, tool['id'])
        tool_id_map[tool['name']] = tool['id']
    logging.debug(f"[V2] Linked {len(tools_list)} tools to attempt")

    # Create container for isolated execution
    container = ContainerManager(image=image, runtime=runtime)
    container_id = container.start(worktree_path)
    logging.info(f"[V2] Started container {container_id[:12]}")

    try:
        # Initialize tool executor and create tool functions
        tools_executor = ToolExecutor(container)
        tool_functions = create_v2_tool_functions(tools_executor)

        # Track tool calls and activity
        tool_calls_made = []
        last_activity_time = [time.time()]

        def after_call(tool, tool_call, tool_result):
            """Record each tool call"""
            # Check timeout (2 minutes)
            elapsed = time.time() - last_activity_time[0]
            if elapsed > 120:
                logging.error(f"[V2 Attempt {attempt_id}] Timeout: No activity for {elapsed:.1f}s")
                raise LLMTimeoutError(f"No activity for {elapsed:.1f}s (timeout: 120s)")

            last_activity_time[0] = time.time()

            # Get tool name
            tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))

            # Log
            args_str = str(tool_call.arguments)[:100]
            logging.info(f"[V2]   Tool {len(tool_calls_made)+1}/{max_tools}: {tool_name}({args_str})")

            # Record tool call
            tool_id = tool_id_map.get(tool_name)
            if tool_id:
                create_tool_call_v2(
                    db, attempt_id, len(tool_calls_made) + 1, tool_id, tool_name,
                    input_data=json.dumps(tool_call.arguments),
                    output=str(tool_result.output)
                )

            tool_calls_made.append({'name': tool_name, 'args': tool_call.arguments})

            # Check tool limit
            if len(tool_calls_made) >= max_tools:
                logging.warning(f"[V2 Attempt {attempt_id}] Tool limit ({max_tools}) reached")
                raise ToolLimitExceeded(f"Tool limit of {max_tools} exceeded")

        try:
            # Call LLM with tools
            logging.info(f"[V2] Calling LLM {model_id}...")
            model = llm.get_model(model_id)

            # Get validation steps to include in prompt
            validation_steps = get_validation_steps_v2(db, goal_id)
            validation_text = ""
            if validation_steps:
                validation_text = "\n\n**Success criteria (validation commands):**\n"
                validation_text += "Your changes will be validated by running these commands:\n"
                for step in validation_steps:
                    validation_text += f"- `{step['command']}`\n"
                validation_text += "\n**IMPORTANT:** Run these validation commands yourself using the appropriate tools (e.g., bazel_build, bazel_test).\n"
                validation_text += "Once ALL validation commands pass, STOP making changes and explain that you've completed the goal.\n"
                validation_text += "Do NOT continue making unnecessary changes or writing documentation after validation passes."

            system_prompt = f"""You are an autonomous coding agent working on a specific goal in a git repository.

You have access to tools to read files, search code, modify files, and run Bazel commands.

**Important constraints:**
- You have {max_tools} tool calls to achieve this goal
- Use tools efficiently to stay within the limit
- For Bazel projects: use MODULE.bazel with bzlmod, NOT WORKSPACE files{validation_text}

Your task is to work towards achieving the goal. You should:
1. Explore the repository to understand its structure
2. Make necessary changes to achieve the goal
3. Validate your changes by running the validation commands listed above
4. STOP once validation passes - do not make unnecessary changes

When you have successfully achieved the goal (or determined it cannot be achieved), explain your final status clearly and STOP."""

            chain = model.chain(
                goal['goal_text'],
                system=system_prompt,
                tools=tool_functions,
                after_call=after_call
            )

            response_text = chain.text()
            logging.info(f"[V2] LLM completed - {len(tool_calls_made)} tools used")

            # Check for uncommitted changes
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                check=True
            )
            has_changes = bool(status_result.stdout.strip())

            if has_changes:
                # Commit all changes
                logging.info(f"[V2] Committing changes made by LLM...")
                subprocess.run(['git', 'add', '-A'], cwd=worktree_path, check=True, capture_output=True)

                commit_msg = f"""[V2] Goal {goal_id}: {goal['goal_text']}

Attempt {attempt_id} by {model_id}

Prompt sent to LLM:
{prompt}
"""
                subprocess.run(
                    ['git', 'commit', '-m', commit_msg],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True
                )
                logging.info(f"[V2] Changes committed")
            else:
                logging.info(f"[V2] No changes to commit")

            # Get final commit SHA
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                check=True
            )
            end_sha = result.stdout.strip()

            # Get diff between start and end
            diff_result = subprocess.run(
                ['git', 'diff', goal_branch_sha, end_sha],
                cwd=worktree_path,
                capture_output=True,
                text=True
            )
            diff = diff_result.stdout

            status = "completed"  # LLM finished
            status_detail = f"Used {len(tool_calls_made)} tools"

        except LLMTimeoutError as e:
            logging.error(f"[V2] Timeout: {e}")
            end_sha = goal_branch_sha
            diff = ""
            status = "timeout"
            status_detail = str(e)
        except ToolLimitExceeded as e:
            logging.error(f"[V2] Tool limit exceeded: {e}")
            end_sha = goal_branch_sha
            diff = ""
            status = "tool_limit"
            status_detail = str(e)
        except Exception as e:
            logging.error(f"[V2] Error: {e}")
            end_sha = goal_branch_sha
            diff = ""
            status = "error"
            status_detail = str(e)
    finally:
        # Stop the container
        container.stop()

    # Create attempt result
    result_id = create_attempt_result_v2(
        db, attempt_id, end_sha, diff, status, status_detail
    )
    logging.info(f"[V2] Created attempt result {result_id}: {status}")

    # Run validation if goal has validation steps (use container for consistency)
    validation_steps = get_validation_steps_v2(db, goal_id)
    validation_passed = None

    if validation_steps:
        logging.info(f"[V2] Running {len(validation_steps)} validation steps in container")
        all_passed = True

        # Restart container for validation (it may have been stopped)
        container_restarted = False
        try:
            # Check if container is still running
            check_result = subprocess.run(
                [runtime if runtime else 'container', 'inspect', '-f', '{{.State.Running}}', container_id],
                capture_output=True,
                text=True
            )
            if check_result.returncode != 0 or check_result.stdout.strip() != 'true':
                container_id = container.start(worktree_path)
                container_restarted = True
                logging.info(f"[V2] Restarted container for validation")
        except:
            container_id = container.start(worktree_path)
            container_restarted = True
            logging.info(f"[V2] Started new container for validation")

        for step in validation_steps:
            # Execute validation command in container
            val_result = container.exec(step['command'], timeout=300)

            create_validation_run_v2(
                db,
                validation_step_id=step['id'],
                attempt_result_id=result_id,
                exit_code=val_result.returncode,
                output=val_result.stdout + val_result.stderr
            )

            if val_result.returncode != 0:
                all_passed = False
                logging.warning(f"[V2]   Validation FAILED: {step['command']}")
            else:
                logging.info(f"[V2]   Validation passed: {step['command']}")

        if container_restarted:
            container.stop()

        validation_passed = all_passed
        logging.info(f"[V2] Validation: {'PASSED' if validation_passed else 'FAILED'}")

        # Update status to "success" if validation passed
        if validation_passed and status == "completed":
            status = "success"
            # Update the attempt_result in the database
            db.execute(
                "UPDATE attempt_result SET status = ? WHERE id = ?",
                (status, result_id)
            )
            db.commit()
            logging.info(f"[V2] Updated result status to 'success'")

    # Merge attempt branch into goal branch on success
    if status == "success" and validation_passed:
        try:
            logging.info(f"[V2] Merging {attempt_branch_name} into {goal_branch_name}")

            # Checkout goal branch in main repo
            subprocess.run(
                ['git', 'checkout', goal_branch_name],
                cwd=repo_path,
                check=True,
                capture_output=True
            )

            # Merge attempt branch with --no-ff to preserve history
            subprocess.run(
                ['git', 'merge', '--no-ff', '-m', f'Merge attempt {attempt_id} for goal {goal_id}', attempt_branch_name],
                cwd=repo_path,
                check=True,
                capture_output=True
            )

            # Get merge commit SHA
            merge_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            merge_commit_sha = merge_result.stdout.strip()
            logging.info(f"[V2] ✓ Merged successfully (commit: {merge_commit_sha[:8]})")

            # Record merge in database
            # Get branch IDs
            goal_branch_result = db.execute(
                "SELECT id FROM branch WHERE name = ? AND session_id = ?",
                (goal_branch_name, session_id)
            ).fetchone()

            if goal_branch_result:
                goal_branch_id = goal_branch_result[0]
                create_merge_v2(
                    db,
                    from_branch_id=attempt_branch_id,
                    from_commit_sha=end_sha,
                    to_branch_id=goal_branch_id,
                    to_commit_sha=goal_branch_sha,
                    result_commit_sha=merge_commit_sha
                )
                logging.info(f"[V2] Recorded merge in database")

        except subprocess.CalledProcessError as e:
            logging.error(f"[V2] Failed to merge: {e}")
            status = "error"
            # Don't fail the entire attempt, just log the error

    # Cleanup: remove worktree
    try:
        logging.info(f"[V2] Cleaning up worktree: {worktree_path}")
        subprocess.run(
            ['git', 'worktree', 'remove', worktree_path, '--force'],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        logging.info(f"[V2] ✓ Worktree removed")
    except subprocess.CalledProcessError as e:
        logging.warning(f"[V2] Failed to remove worktree: {e}")

    success = (status == "success") and (validation_passed is True)

    return {
        'success': success,
        'attempt_id': attempt_id,
        'result_id': result_id,
        'validation_passed': validation_passed,
        'worktree_path': worktree_path,
        'goal_branch': goal_branch_name
    }


def breakdown_goal_v2(db, session_id, goal_id, failed_attempt_id, model_id, repo_path):
    """
    Break down a goal into sub-goals using an LLM (breakdown attempt).

    Creates a proper breakdown attempt (with attempt_type='breakdown') that:
    1. Creates branch/worktree for isolation
    2. Records the breakdown attempt with read-only tools
    3. Simulates LLM analysis (would call real LLM in production)
    4. Records tool calls made during breakdown
    5. Creates attempt_result for the breakdown
    6. Creates sub-goals linked to the breakdown attempt

    Args:
        db: Database connection
        session_id: Session ID
        goal_id: Goal that failed
        failed_attempt_id: The attempt that failed
        model_id: Model to use for breakdown
        repo_path: Path to repository

    Returns:
        Tuple of (breakdown_attempt_id, sub_goal_ids)
    """
    goal = get_goal_v2(db, goal_id)
    failed_attempt = get_attempt_v2(db, failed_attempt_id)

    if not goal or not failed_attempt:
        raise ValueError("Goal or failed attempt not found")

    logging.info(f"[V2 Breakdown] Breaking down goal {goal_id} after attempt {failed_attempt_id} failed")

    # Get current commit SHA
    result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                          cwd=repo_path, capture_output=True,
                          text=True, check=True)
    current_sha = result.stdout.strip()

    # Get next attempt ID for unique paths
    cursor = db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM attempt")
    next_attempt_id = cursor.fetchone()[0]

    # Create unique branch name for breakdown (never reused - includes timestamp)
    timestamp = int(time.time())
    branch_name = f"s{session_id}-g{goal_id}-a{next_attempt_id}-{timestamp}-breakdown"

    # Create branch (always create new, never reuse)
    branch_id = create_branch_v2(
        db, session_id,
        name=branch_name,
        parent_commit_sha=current_sha,
        reason=f"Breakdown for goal {goal_id}",
        created_by_goal_id=goal_id
    )

    # Create worktree path (globally unique)
    worktree_path = f"{repo_path}/worktrees/s{session_id}-g{goal_id}-a{next_attempt_id}-{timestamp}-breakdown"

    # Create git worktree with new branch
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)
    subprocess.run(
        ['git', 'worktree', 'add', '-b', branch_name, worktree_path, current_sha],
        cwd=repo_path,
        check=True,
        capture_output=True
    )
    logging.info(f"[V2 Breakdown] Created git worktree: {worktree_path} (branch: {branch_name})")

    # Track worktree in database
    worktree_id = create_worktree_v2(
        db, branch_id, worktree_path, current_sha,
        reason=f"Breakdown attempt for goal {goal_id}"
    )

    # Create breakdown attempt
    prompt = f"Analyze goal '{goal['goal_text']}' and break it down into sub-goals."
    breakdown_attempt_id = create_attempt_v2(
        db, goal_id, worktree_id, current_sha,
        prompt, model_id, attempt_type='breakdown'
    )

    logging.info(f"[V2 Breakdown] Created breakdown attempt {breakdown_attempt_id}")

    # Get read-only tools for this session
    all_tools = get_tools_for_session_v2(db, session_id)
    # In real implementation, filter to read-only tools (list_directory, read_file, etc.)
    # For simulation, just use first 2 tools as "read-only"
    read_only_tools = all_tools[:min(2, len(all_tools))]

    # Link read-only tools to breakdown attempt
    for tool in read_only_tools:
        create_attempt_tool_v2(db, breakdown_attempt_id, tool['id'])

    # Simulate LLM breakdown with read-only tools
    # In real implementation, this would call the LLM with read-only tools
    # and parse the response to extract sub-goals
    tool_call_count = 0
    for tool in read_only_tools:
        tool_call_count += 1
        create_tool_call_v2(
            db, breakdown_attempt_id, tool_call_count, tool['id'],
            tool['name'],
            input_data='{"simulated": "input"}',
            output="Simulated output from breakdown analysis"
        )

    # Simulated sub-goals (in real implementation, parsed from LLM response)
    sub_goals_data = [
        {
            "goal_text": f"Sub-goal 1 of: {goal['goal_text'][:40]}...",
            "validation_commands": []
        },
        {
            "goal_text": f"Sub-goal 2 of: {goal['goal_text'][:40]}...",
            "validation_commands": []
        },
        {
            "goal_text": f"Sub-goal 3 of: {goal['goal_text'][:40]}...",
            "validation_commands": []
        }
    ]

    # Create attempt_result for breakdown (status='success' since we got sub-goals)
    breakdown_result_id = create_attempt_result_v2(
        db, breakdown_attempt_id,
        end_commit_sha=current_sha,  # No changes during breakdown
        diff="",  # No diff for read-only breakdown
        status="success",
        status_detail=f"Broke down into {len(sub_goals_data)} sub-goals"
    )

    logging.info(f"[V2 Breakdown] Created breakdown result {breakdown_result_id}")

    # Create sub-goals linked to breakdown attempt
    sub_goal_ids = []
    for i, sub_goal_data in enumerate(sub_goals_data, 1):
        sub_goal_id = create_goal_v2(
            db,
            session_id=session_id,
            goal_text=sub_goal_data['goal_text'],
            parent_goal_id=goal_id,
            order_num=i,
            source='breakdown',
            created_by_attempt_id=breakdown_attempt_id  # Link to breakdown attempt!
        )

        # Create validation steps if specified
        for j, val_cmd in enumerate(sub_goal_data.get('validation_commands', []), 1):
            create_validation_step_v2(
                db,
                goal_id=sub_goal_id,
                order_num=j,
                command=val_cmd,
                source='breakdown'
            )

        sub_goal_ids.append(sub_goal_id)
        logging.info(f"[V2 Breakdown] Created sub-goal {i}/{len(sub_goals_data)}: {sub_goal_data['goal_text'][:50]}")

    return (breakdown_attempt_id, sub_goal_ids)


def work_on_goal_v2_recursive(db, session_id, goal_id, repo_path,
                               model_a, model_b, current_model=None,
                               max_decompositions=5, decomposition_count=0,
                               depth=0, max_depth=5, max_tools=50,
                               image='shots-on-goal:latest', runtime='container'):
    """
    Recursively work on a goal with ping-pong model alternation and breakdown.

    Flow:
    1. Try with current model (or model_a if first attempt)
    2. If succeeds -> done
    3. If fails -> other model performs breakdown
    4. Other model works on sub-goals recursively

    Args:
        db: Database connection
        session_id: Session ID
        goal_id: Goal to work on
        repo_path: Repository path
        model_a: First model
        model_b: Second model
        current_model: Currently active model (None means start with model_a)
        max_decompositions: Max decompositions allowed
        decomposition_count: Current decomposition depth
        depth: Current recursion depth
        max_depth: Max recursion depth
        max_tools: Max tool calls per attempt

    Returns:
        bool: True if goal succeeded, False otherwise
    """
    indent = "  " * depth

    # Check limits
    if decomposition_count >= max_decompositions:
        logging.warning(f"{indent}[V2] Max decompositions ({max_decompositions}) reached for goal {goal_id}")
        return False

    if depth >= max_depth:
        logging.warning(f"{indent}[V2] Max depth ({max_depth}) reached for goal {goal_id}")
        return False

    # Determine which model to use
    if current_model is None:
        model_to_use = model_a
        other_model = model_b
    else:
        model_to_use = current_model
        other_model = model_b if current_model == model_a else model_a

    goal = get_goal_v2(db, goal_id)
    logging.info(f"{indent}[V2] Working on goal {goal_id}: {goal['goal_text'][:50]}")
    logging.info(f"{indent}[V2] Using model: {model_to_use}")

    # Attempt to complete the goal
    try:
        result = work_on_goal_v2_simple(
            db,
            session_id=session_id,
            goal_id=goal_id,
            repo_path=repo_path,
            model_id=model_to_use,
            attempt_type='implementation',
            max_tools=max_tools,
            image=image,
            runtime=runtime
        )

        if result['success']:
            logging.info(f"{indent}[V2] ✓ Goal {goal_id} succeeded")
            return True
        else:
            logging.info(f"{indent}[V2] ✗ Goal {goal_id} failed")

    except Exception as e:
        logging.error(f"{indent}[V2] ✗ Goal {goal_id} failed with error: {e}")
        result = {'success': False, 'attempt_id': None}

    # Attempt failed - break down with other model
    if result.get('attempt_id'):
        failed_attempt_id = result['attempt_id']
    else:
        logging.error(f"{indent}[V2] No attempt ID available for breakdown, cannot continue")
        return False

    logging.info(f"{indent}[V2] Breaking down goal {goal_id} with {other_model}")

    # Perform breakdown
    try:
        breakdown_attempt_id, sub_goal_ids = breakdown_goal_v2(
            db,
            session_id=session_id,
            goal_id=goal_id,
            failed_attempt_id=failed_attempt_id,
            model_id=other_model,
            repo_path=repo_path
        )
        logging.info(f"{indent}[V2] Breakdown attempt {breakdown_attempt_id} completed")
    except Exception as e:
        logging.error(f"{indent}[V2] Breakdown failed: {e}")
        return False

    if not sub_goal_ids:
        logging.error(f"{indent}[V2] No sub-goals created during breakdown")
        return False

    logging.info(f"{indent}[V2] Created {len(sub_goal_ids)} sub-goals, working on them with {other_model}")

    # Work on sub-goals recursively with the model that did the breakdown
    all_succeeded = True
    for i, sub_goal_id in enumerate(sub_goal_ids, 1):
        logging.info(f"{indent}[V2] Sub-goal {i}/{len(sub_goal_ids)}")

        success = work_on_goal_v2_recursive(
            db,
            session_id=session_id,
            goal_id=sub_goal_id,
            repo_path=repo_path,
            model_a=model_a,
            model_b=model_b,
            current_model=other_model,  # Model that broke down continues with sub-goals
            max_decompositions=max_decompositions,
            decomposition_count=decomposition_count + 1,
            depth=depth + 1,
            max_depth=max_depth,
            max_tools=max_tools,
            image=image,
            runtime=runtime
        )

        if not success:
            all_succeeded = False
            logging.warning(f"{indent}[V2] Sub-goal {sub_goal_id} failed")

    if all_succeeded:
        logging.info(f"{indent}[V2] ✓ All sub-goals succeeded for goal {goal_id}")
    else:
        logging.warning(f"{indent}[V2] ✗ Some sub-goals failed for goal {goal_id}")

    return all_succeeded


# ============================================================================


# ============================================================================
# Container Management
# ============================================================================

def detect_container_runtime():
    """
    Detect which container runtime is available.
    Returns 'container' or 'docker', or raises RuntimeError if neither found.
    """
    for runtime in ['container', 'docker']:
        try:
            subprocess.run(
                [runtime, '--version'],
                check=True,
                capture_output=True,
                timeout=5
            )
            return runtime
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue

    raise RuntimeError("No container runtime found. Please install 'container' or 'docker'.")


class ContainerManager:
    """Manages container lifecycle for executing attempts"""

    def __init__(self, image="ubuntu:22.04", runtime="container"):
        """
        Initialize container manager.

        Args:
            image: Container image to use
            runtime: Container runtime command ('container' or 'docker')
        """
        self.image = image
        self.runtime = runtime
        self.container_id = None

    def start(self, worktree_path, shared_cache_dir=None):
        """
        Start a container with worktree mounted.

        Args:
            worktree_path: Path to worktree to mount at /workspace
            shared_cache_dir: Optional shared cache directory (e.g., for bazel)

        Returns:
            container_id
        """
        # Convert to absolute path for container mount
        abs_worktree_path = os.path.abspath(worktree_path)

        cmd = [
            self.runtime, 'run',
            '-d',  # Detached
            '--rm',  # Auto-remove when stopped
            '-v', f'{abs_worktree_path}:/workspace',
            '-w', '/workspace',
        ]

        # Add shared cache mount if specified
        if shared_cache_dir:
            cmd.extend(['-v', f'{shared_cache_dir}:/root/.cache'])

        # Image and command
        cmd.extend([self.image, 'sleep', 'infinity'])

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        self.container_id = result.stdout.strip()
        return self.container_id

    def exec(self, command, timeout=None):
        """
        Execute a command in the container.

        Args:
            command: Shell command to execute
            timeout: Optional timeout in seconds

        Returns:
            subprocess.CompletedProcess with stdout, stderr, returncode
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        cmd = [
            self.runtime, 'exec',
            '-w', '/workspace',
            self.container_id,
            'bash', '-c', command
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return result

    def stop(self):
        """Stop and remove the container"""
        if self.container_id:
            subprocess.run(
                [self.runtime, 'stop', self.container_id],
                check=False,  # Don't fail if already stopped
                capture_output=True
            )
            self.container_id = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop container"""
        self.stop()
        return False


# ============================================================================
# Tool Execution
# ============================================================================

class ToolExecutor:
    """Executes tools in a container with restricted, safe operations"""

    def __init__(self, container):
        """
        Initialize with a container manager.

        Args:
            container: ContainerManager instance
        """
        self.container = container

    # Bazel Tools

    def bazel_build(self, targets="//...", flags=None):
        """
        Build Bazel targets.

        Args:
            targets: Bazel target pattern (default: //...)
            flags: Optional list of additional bazel flags

        Returns:
            dict with 'success', 'stdout', 'stderr', 'exit_code'
        """
        cmd = f"bazel build {targets}"
        if flags:
            cmd += " " + " ".join(flags)

        result = self.container.exec(cmd, timeout=600)

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }

    def bazel_test(self, targets="//...", flags=None):
        """
        Run Bazel tests.

        Args:
            targets: Bazel test target pattern (default: //...)
            flags: Optional list of additional bazel flags

        Returns:
            dict with 'success', 'stdout', 'stderr', 'exit_code'
        """
        cmd = f"bazel test {targets}"
        if flags:
            cmd += " " + " ".join(flags)

        result = self.container.exec(cmd, timeout=600)

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }

    def bazel_query(self, query):
        """
        Query the Bazel build graph.

        Args:
            query: Bazel query expression

        Returns:
            dict with 'success', 'stdout', 'stderr', 'exit_code'
        """
        # Escape quotes in query
        query_escaped = query.replace('"', '\\"')
        cmd = f'bazel query "{query_escaped}"'

        result = self.container.exec(cmd, timeout=60)

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.returncode
        }

    # File Operations

    def read_file(self, path):
        """
        Read a file from the workspace.

        Args:
            path: Path to file (relative to workspace root)

        Returns:
            dict with 'success', 'content', 'error'
        """
        # Use cat to read the file
        result = self.container.exec(f"cat {path}", timeout=30)

        if result.returncode == 0:
            return {
                'success': True,
                'content': result.stdout,
                'error': None
            }
        else:
            return {
                'success': False,
                'content': None,
                'error': result.stderr
            }

    def read_multiple_files(self, paths):
        """
        Read multiple files from the workspace at once.

        Args:
            paths: List of file paths relative to workspace root

        Returns:
            dict mapping path to result dict with 'success', 'content', 'error'
        """
        results = {}
        for path in paths:
            results[path] = self.read_file(path)
        return results

    def write_file(self, path, content):
        """
        Write content to a file in the workspace.

        Args:
            path: Path to file (relative to workspace root)
            content: Content to write

        Returns:
            dict with 'success', 'error'
        """
        # Escape content for shell
        content_escaped = content.replace("'", "'\\''")

        # Create directory if needed, then write file
        cmd = f"mkdir -p $(dirname {path}) && printf '%s' '{content_escaped}' > {path}"
        result = self.container.exec(cmd, timeout=30)

        return {
            'success': result.returncode == 0,
            'error': result.stderr if result.returncode != 0 else None
        }

    def find_replace_in_file(self, path, old_text, new_text):
        """
        Find and replace text in a file. Requires exactly one match.

        Args:
            path: Path to file (relative to workspace root)
            old_text: Text to find (must match exactly once)
            new_text: Text to replace with

        Returns:
            dict with 'success', 'error'
        """
        # First, read the file to check match count
        read_result = self.read_file(path)
        if not read_result['success']:
            return {
                'success': False,
                'error': f"Could not read file: {read_result['error']}"
            }

        content = read_result['content']
        match_count = content.count(old_text)

        if match_count == 0:
            return {
                'success': False,
                'error': f"No matches found for old_text in {path}"
            }
        elif match_count > 1:
            return {
                'success': False,
                'error': f"Found {match_count} matches for old_text in {path}, expected exactly 1"
            }

        # Exactly one match - do the replacement
        new_content = content.replace(old_text, new_text, 1)
        write_result = self.write_file(path, new_content)

        if write_result['success']:
            return {
                'success': True,
                'error': None
            }
        else:
            return {
                'success': False,
                'error': f"Failed to write file: {write_result['error']}"
            }

    def list_directory(self, path="."):
        """
        List files in a directory.

        Args:
            path: Directory path (default: current directory)

        Returns:
            dict with 'success', 'files', 'error'
        """
        result = self.container.exec(f"ls -1 {path}", timeout=30)

        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split('\n') if f]
            return {
                'success': True,
                'files': files,
                'error': None
            }
        else:
            return {
                'success': False,
                'files': [],
                'error': result.stderr
            }

    # Code Search Tools

    def ripgrep(self, pattern, path=".", glob=None, ignore_case=False):
        """
        Search code using ripgrep.

        Args:
            pattern: Search pattern (regex)
            path: Path to search in (default: .)
            glob: Optional glob pattern to filter files
            ignore_case: Case-insensitive search

        Returns:
            dict with 'success', 'matches', 'stdout', 'stderr'
        """
        cmd = f"rg --json '{pattern}' {path}"

        if glob:
            cmd += f" --glob '{glob}'"

        if ignore_case:
            cmd += " -i"

        result = self.container.exec(cmd, timeout=60)

        # ripgrep returns 1 if no matches found, which is not an error
        success = result.returncode in [0, 1]

        return {
            'success': success,
            'matches': result.stdout,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    def find_files(self, pattern, path="."):
        """
        Find files by name pattern.

        Args:
            pattern: Filename pattern (glob)
            path: Path to search in (default: .)

        Returns:
            dict with 'success', 'files', 'error'
        """
        cmd = f"find {path} -name '{pattern}'"
        result = self.container.exec(cmd, timeout=60)

        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split('\n') if f]
            return {
                'success': True,
                'files': files,
                'error': None
            }
        else:
            return {
                'success': False,
                'files': [],
                'error': result.stderr
            }


# ============================================================================
# Work Loop
# ============================================================================

def create_tool_functions(tools_executor):
    """
    Create LLM tool functions that wrap ToolExecutor methods.

    Args:
        tools_executor: ToolExecutor instance

    Returns:
        List of tool functions with docstrings
    """

    def read_file(path: str) -> str:
        """
        Read a file from the workspace.

        Args:
            path: Path to file relative to workspace root

        Returns:
            File contents as a string
        """
        result = tools_executor.read_file(path)
        if result['success']:
            return result['content']
        else:
            return f"ERROR: {result['error']}"

    def read_multiple_files(paths: list) -> str:
        """
        Read multiple files at once (more efficient than calling read_file multiple times).

        Args:
            paths: List of file paths relative to workspace root

        Returns:
            Formatted string with contents of each file, separated by headers
        """
        results = tools_executor.read_multiple_files(paths)
        output = []
        for path, result in results.items():
            if result['success']:
                output.append(f"=== {path} ===\n{result['content']}")
            else:
                output.append(f"=== {path} ===\nERROR: {result['error']}")
        return "\n\n".join(output) if output else "No files read"

    def write_file(path: str, content: str) -> str:
        """
        Write content to a file in the workspace.

        Args:
            path: Path to file relative to workspace root
            content: Content to write to the file

        Returns:
            Success message or error
        """
        result = tools_executor.write_file(path, content)
        if result['success']:
            return f"Successfully wrote to {path}"
        else:
            return f"ERROR: {result['error']}"

    def write_multiple_files(files: dict) -> str:
        """
        Write multiple files at once (more efficient than calling write_file multiple times).

        Args:
            files: Dict mapping file paths to content strings
                   Example: {"src/foo.py": "content1", "src/bar.py": "content2"}

        Returns:
            Summary of files written or errors
        """
        results = tools_executor.write_multiple_files(files)
        output = []
        success_count = 0
        for path, result in results.items():
            if result['success']:
                success_count += 1
                output.append(f"✓ {path}")
            else:
                output.append(f"✗ {path}: {result['error']}")

        summary = f"Wrote {success_count}/{len(results)} files successfully"
        if output:
            return f"{summary}\n" + "\n".join(output)
        return summary

    def find_replace_in_file(path: str, old_text: str, new_text: str) -> str:
        """
        Find and replace text in a file. Requires exactly one match.

        Args:
            path: Path to file relative to workspace root
            old_text: Text to find (must match exactly once)
            new_text: Text to replace with

        Returns:
            Success message or error
        """
        result = tools_executor.find_replace_in_file(path, old_text, new_text)
        if result['success']:
            return f"Successfully replaced text in {path}"
        else:
            return f"ERROR: {result['error']}"

    def list_directory(path: str = ".") -> str:
        """
        List files in a directory.

        Args:
            path: Directory path (default: current directory)

        Returns:
            Newline-separated list of files
        """
        result = tools_executor.list_directory(path)
        if result['success']:
            return "\n".join(result['files'])
        else:
            return f"ERROR: {result['error']}"

    def find_files(pattern: str, path: str = ".") -> str:
        """
        Find files by name pattern.

        Args:
            pattern: Filename pattern (glob style, e.g., "*.py" or "BUILD*")
            path: Path to search in (default: current directory)

        Returns:
            Newline-separated list of matching file paths
        """
        result = tools_executor.find_files(pattern, path)
        if result['success']:
            return "\n".join(result['files']) if result['files'] else "No files found"
        else:
            return f"ERROR: {result['error']}"

    def ripgrep(pattern: str, path: str = ".", glob: str = None, ignore_case: bool = False) -> str:
        """
        Search code using ripgrep.

        Args:
            pattern: Search pattern (regex)
            path: Path to search in (default: current directory)
            glob: Optional glob pattern to filter files (e.g., "*.py")
            ignore_case: Case-insensitive search (default: false)

        Returns:
            Search results in JSON format
        """
        result = tools_executor.ripgrep(pattern, path, glob, ignore_case)
        if result['success']:
            return result['stdout'] if result['stdout'] else "No matches found"
        else:
            return f"ERROR: {result['stderr']}"

    def bazel_build(targets: str = "//...", flags: str = None) -> str:
        """
        Build Bazel targets.

        Args:
            targets: Bazel target pattern (default: //...)
            flags: Optional space-separated bazel flags

        Returns:
            Build output or error message
        """
        flag_list = flags.split() if flags else None
        result = tools_executor.bazel_build(targets, flag_list)

        output = []
        if result['stdout']:
            output.append(result['stdout'])
        if result['stderr']:
            output.append(result['stderr'])

        if result['success']:
            return "\n".join(output) or "Build succeeded"
        else:
            return f"Build failed (exit code {result['exit_code']}):\n" + "\n".join(output)

    def bazel_test(targets: str = "//...", flags: str = None) -> str:
        """
        Run Bazel tests.

        Args:
            targets: Bazel test target pattern (default: //...)
            flags: Optional space-separated bazel flags

        Returns:
            Test output or error message
        """
        flag_list = flags.split() if flags else None
        result = tools_executor.bazel_test(targets, flag_list)

        output = []
        if result['stdout']:
            output.append(result['stdout'])
        if result['stderr']:
            output.append(result['stderr'])

        if result['success']:
            return "\n".join(output) or "Tests passed"
        else:
            return f"Tests failed (exit code {result['exit_code']}):\n" + "\n".join(output)

    def bazel_query(query: str) -> str:
        """
        Query the Bazel build graph.

        Args:
            query: Bazel query expression (e.g., "//..." or "deps(//pkg:target)")

        Returns:
            Query results or error message
        """
        result = tools_executor.bazel_query(query)

        output = []
        if result['stdout']:
            output.append(result['stdout'])
        if result['stderr']:
            output.append(result['stderr'])

        if result['success']:
            return "\n".join(output) or "Query succeeded"
        else:
            return f"Query failed (exit code {result['exit_code']}):\n" + "\n".join(output)

    return [
        read_file,
        read_multiple_files,
        write_file,
        write_multiple_files,
        find_replace_in_file,
        list_directory,
        find_files,
        ripgrep,
        bazel_build,
        bazel_test,
        bazel_query,
    ]


class ToolLimitExceeded(Exception):
    """Raised when tool call limit is exceeded"""
    pass





# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description="Shots on Goal - Autonomous goal-driven code migration"
    )

    # Main command arguments
    parser.add_argument(
        'goal',
        nargs='?',
        help='Goal description (e.g., "Migrate to Bazel")'
    )
    parser.add_argument(
        'repo_path',
        nargs='?',
        help='Path to git repository'
    )

    # Configuration options
    parser.add_argument(
        '--model-a',
        default='openrouter/anthropic/claude-sonnet-4.5',
        help='First model for collaborative peer review (default: claude-sonnet-4.5)'
    )
    parser.add_argument(
        '--model-b',
        default='openrouter/openai/gpt-5-codex',
        help='Second model for collaborative peer review (default: gpt-5-codex)'
    )
    parser.add_argument(
        '--image',
        default='shots-on-goal:latest',
        help='Container image to use (default: shots-on-goal:latest)'
    )
    parser.add_argument(
        '--max-tools',
        type=int,
        default=50,
        help='Maximum number of tool calls per attempt (default: 50)'
    )
    parser.add_argument(
        '--max-goal-breakdowns',
        type=int,
        default=5,
        help='Maximum number of goal breakdowns before giving up (default: 5)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    parser.add_argument(
        '--validation',
        action='append',
        help='Validation command(s) to run (can be specified multiple times)'
    )

    args = parser.parse_args()

    # Update logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate required arguments
    if not args.goal or not args.repo_path:
        parser.print_help()
        sys.exit(1)

    # Validate repo path
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        logging.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)

    if not (repo_path / ".git").exists():
        logging.error(f"Not a git repository: {repo_path}")
        sys.exit(1)

    # Detect default branch
    try:
        base_branch = detect_default_branch(str(repo_path))
        logging.info(f"Detected base branch: {base_branch}")
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    # ========================================================================
    # Execute with V2 Schema
    # ========================================================================
    logging.info("=" * 80)
    logging.info("Shots on Goal - Autonomous Code Migration")
    logging.info("=" * 80)

    # Create V2 database
    db_path = f"shots-on-goal-v2-{int(time.time())}.db"
    db = init_database_v2(db_path)
    logging.info(f"Created database: {db_path}")

    # Create V2 session
    session_id = create_session_v2(
        db,
        initial_goal=args.goal,
        model_a=args.model_a,
        model_b=args.model_b,
        flags={'max_tools': args.max_tools, 'max_goal_breakdowns': args.max_goal_breakdowns},
        repo_path=str(repo_path),
        base_branch=base_branch
    )
    logging.info(f"Created V2 session (ID: {session_id})")

    # Create tools in V2 schema
    tool_definitions = [
        ('read_file', 'Read a file from the workspace'),
        ('read_multiple_files', 'Read multiple files at once (more efficient than calling read_file multiple times)'),
        ('write_file', 'Write content to a file'),
        ('write_multiple_files', 'Write multiple files at once (more efficient than calling write_file multiple times)'),
        ('find_replace_in_file', 'Find and replace text in a file'),
        ('list_directory', 'List files in a directory'),
        ('find_files', 'Find files by name pattern'),
        ('ripgrep', 'Search code using ripgrep'),
        ('bazel_build', 'Build Bazel targets'),
        ('bazel_test', 'Run Bazel tests'),
        ('bazel_query', 'Query the Bazel build graph'),
    ]

    for name, description in tool_definitions:
        create_tool_v2(db, session_id, name, description)
    logging.info(f"Created {len(tool_definitions)} tools")

    # Create root goal with validation commands
    root_goal_id = create_goal_v2(
        db,
        session_id=session_id,
        goal_text=args.goal,
        source='cli'
    )
    logging.info(f"Created root goal (ID: {root_goal_id}): {args.goal}")

    # Add validation steps if provided
    if args.validation:
        for i, val_cmd in enumerate(args.validation, 1):
            create_validation_step_v2(
                db,
                goal_id=root_goal_id,
                order_num=i,
                command=val_cmd,
                source='cli'
            )
        logging.info(f"Added {len(args.validation)} validation steps")

    # Start working on the root goal
    logging.info("=" * 80)
    logging.info("Starting work on goal with V2 orchestration...")
    logging.info("=" * 80)

    try:
        success = work_on_goal_v2_recursive(
            db=db,
            session_id=session_id,
            goal_id=root_goal_id,
            repo_path=str(repo_path),
            model_a=args.model_a,
            model_b=args.model_b,
            max_decompositions=args.max_goal_breakdowns,
            max_depth=args.max_goal_breakdowns,  # Use same limit for depth
            max_tools=args.max_tools,
            image=args.image,
            runtime=detect_container_runtime()
        )

        logging.info("=" * 80)
        logging.info("Goal tree completed!")
        logging.info("=" * 80)

        # Get final status
        final_goal = get_goal_v2(db, root_goal_id)
        if success:
            logging.info("✓ Root goal completed successfully!")
        else:
            logging.info("✗ Root goal did not complete successfully")

        # Show goal tree summary
        logging.info("")
        logging.info("--- Goal Tree Summary (V2) ---")
        tree = get_goal_tree_v2(db, root_goal_id)

        def print_tree(node, depth=0):
            indent = "  " * depth
            goal = node['goal']
            attempts = node.get('attempts', [])
            status_icon = "•"
            logging.info(f"{indent}{status_icon} Goal {goal['id']}: {goal['goal_text'][:60]}")
            logging.info(f"{indent}  Attempts: {len(attempts)}")
            for child in node.get('children', []):
                print_tree(child, depth + 1)

        print_tree(tree)

        # Show attempts
        logging.info("")
        logging.info("--- Attempts (V2) ---")
        all_attempts = db.execute("""
            SELECT a.id, a.goal_id, a.model, a.attempt_type,
                   ar.status, w.path as worktree_path
            FROM attempt a
            LEFT JOIN attempt_result ar ON ar.attempt_id = a.id
            LEFT JOIN worktree w ON w.id = a.worktree_id
            ORDER BY a.id
        """).fetchall()

        for attempt in all_attempts:
            status_icon = "✓" if attempt['status'] == 'success' else "✗"
            logging.info(f"{status_icon} Attempt {attempt['id']} (Goal {attempt['goal_id']}, {attempt['model']}, {attempt['attempt_type']}): {attempt['status']}")
            if attempt['worktree_path']:
                logging.info(f"    Worktree: {attempt['worktree_path']}")

    except KeyboardInterrupt:
        logging.info("")
        logging.info("Interrupted by user. Database saved.")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise
    finally:
        db.close()

    return

if __name__ == "__main__":
    main()

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
# Database Setup
# ============================================================================

def init_database(db_path):
    """
    Initialize SQLite database with schema.
    Returns connection object.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row  # Access columns by name

    # Enable foreign key constraints
    db.execute("PRAGMA foreign_keys = ON")

    cursor = db.cursor()

    # Session table (single row per database)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            session_id TEXT NOT NULL,
            description TEXT NOT NULL,
            repo_path TEXT NOT NULL,
            base_branch TEXT DEFAULT 'main',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
    """)

    # Goals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            description TEXT NOT NULL,
            goal_type TEXT DEFAULT 'implementation',
            status TEXT DEFAULT 'pending',
            created_by_goal_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            git_branch TEXT,
            validation_commands TEXT,
            FOREIGN KEY (parent_id) REFERENCES goals(id),
            FOREIGN KEY (created_by_goal_id) REFERENCES goals(id)
        )
    """)

    # Add validation_commands column if it doesn't exist (for existing databases)
    cursor.execute("PRAGMA table_info(goals)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'validation_commands' not in columns:
        cursor.execute("ALTER TABLE goals ADD COLUMN validation_commands TEXT")

    # Attempts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id INTEGER NOT NULL,
            model_id TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            outcome TEXT,
            failure_reason TEXT,
            git_branch TEXT,
            worktree_path TEXT,
            container_id TEXT,
            git_commit_sha TEXT,
            final_commit_sha TEXT,
            FOREIGN KEY (goal_id) REFERENCES goals(id)
        )
    """)

    # Add columns if they don't exist (for existing databases)
    cursor.execute("PRAGMA table_info(attempts)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'final_commit_sha' not in columns:
        cursor.execute("ALTER TABLE attempts ADD COLUMN final_commit_sha TEXT")
    if 'model_id' not in columns:
        cursor.execute("ALTER TABLE attempts ADD COLUMN model_id TEXT")

    # Actions table (tool calls during attempts)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            parameters TEXT,
            result TEXT,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (attempt_id) REFERENCES attempts(id)
        )
    """)

    # Indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_goal ON attempts(goal_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_attempt ON actions(attempt_id)")

    db.commit()
    return db


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
        write_file,
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
                           attempt_type='implementation', max_tools=50):
    """
    V2 orchestration with real LLM integration.

    This function:
    - Creates branch and worktree
    - Tracks attempt in V2 schema
    - Calls real LLM with tools
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

    # Get current commit SHA
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    current_sha = result.stdout.strip()

    # Create or get branch for this goal (include session_id for global uniqueness)
    branch_name = f"s{session_id}-goal-{goal_id}"
    existing_branch = get_branch_by_name_v2(db, session_id, branch_name)

    if existing_branch:
        branch_id = existing_branch['id']
        logging.debug(f"[V2] Using existing branch: {branch_name}")
    else:
        branch_id = create_branch_v2(
            db,
            session_id=session_id,
            name=branch_name,
            parent_commit_sha=current_sha,
            reason=f"Goal {goal_id}",
            created_by_goal_id=goal_id
        )
        logging.info(f"[V2] Created branch: {branch_name}")

    # Get next attempt ID for unique worktree path
    cursor = db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM attempt")
    next_attempt_id = cursor.fetchone()[0]

    # Create worktree path (globally unique: session + goal + attempt)
    worktree_path = f"{repo_path}/worktrees/s{session_id}-g{goal_id}-a{next_attempt_id}"

    # Create git worktree with branch
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)

    # Check if branch exists in git
    branch_check = subprocess.run(
        ['git', 'rev-parse', '--verify', branch_name],
        cwd=repo_path,
        capture_output=True
    )

    if branch_check.returncode == 0:
        # Branch exists, checkout in worktree
        subprocess.run(
            ['git', 'worktree', 'add', worktree_path, branch_name],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
    else:
        # Branch doesn't exist, create it
        subprocess.run(
            ['git', 'worktree', 'add', '-b', branch_name, worktree_path, current_sha],
            cwd=repo_path,
            check=True,
            capture_output=True
        )

    logging.info(f"[V2] Created git worktree: {worktree_path} (branch: {branch_name})")

    # Track worktree in database
    worktree_id = create_worktree_v2(
        db,
        branch_id=branch_id,
        path=worktree_path,
        start_sha=current_sha,
        reason=f"Attempt for goal {goal_id}"
    )

    # Create attempt
    prompt = f"Goal: {goal['goal_text']}\n\nPlease complete this goal."
    attempt_id = create_attempt_v2(
        db,
        goal_id=goal_id,
        worktree_id=worktree_id,
        start_commit_sha=current_sha,
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

    # Initialize tool executor and create tool functions
    tools_executor = WorktreeToolExecutor(worktree_path)
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

        # Determine outcome
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            check=True
        )
        end_sha = result.stdout.strip()

        # Get diff
        diff_result = subprocess.run(
            ['git', 'diff', current_sha, end_sha],
            cwd=worktree_path,
            capture_output=True,
            text=True
        )
        diff = diff_result.stdout

        # Check if changes were made
        has_changes = (end_sha != current_sha) or bool(diff)

        status = "completed"  # LLM finished
        status_detail = f"Used {len(tool_calls_made)} tools"

    except LLMTimeoutError as e:
        logging.error(f"[V2] Timeout: {e}")
        end_sha = current_sha
        diff = ""
        status = "timeout"
        status_detail = str(e)
    except ToolLimitExceeded as e:
        logging.error(f"[V2] Tool limit exceeded: {e}")
        end_sha = current_sha
        diff = ""
        status = "tool_limit"
        status_detail = str(e)
    except Exception as e:
        logging.error(f"[V2] Error: {e}")
        end_sha = current_sha
        diff = ""
        status = "error"
        status_detail = str(e)

    # Create attempt result
    result_id = create_attempt_result_v2(
        db, attempt_id, end_sha, diff, status, status_detail
    )
    logging.info(f"[V2] Created attempt result {result_id}: {status}")

    # Run validation if goal has validation steps
    validation_steps = get_validation_steps_v2(db, goal_id)
    validation_passed = None

    if validation_steps:
        logging.info(f"[V2] Running {len(validation_steps)} validation steps")
        all_passed = True

        for step in validation_steps:
            # Execute validation command
            val_result = subprocess.run(
                step['command'],
                cwd=worktree_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )

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

        validation_passed = all_passed
        logging.info(f"[V2] Validation: {'PASSED' if validation_passed else 'FAILED'}")

    success = (status == "success" or status == "completed") and (validation_passed is not False)

    return {
        'success': success,
        'attempt_id': attempt_id,
        'result_id': result_id,
        'validation_passed': validation_passed,
        'worktree_path': worktree_path
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

    # Create branch for breakdown (globally unique)
    branch_name = f"s{session_id}-goal-{goal_id}-breakdown"
    existing_branch = get_branch_by_name_v2(db, session_id, branch_name)
    if existing_branch:
        branch_id = existing_branch['id']
    else:
        branch_id = create_branch_v2(
            db, session_id,
            name=branch_name,
            parent_commit_sha=current_sha,
            reason=f"Breakdown for goal {goal_id}",
            created_by_goal_id=goal_id
        )

    # Create worktree path (globally unique)
    worktree_path = f"{repo_path}/worktrees/s{session_id}-g{goal_id}-a{next_attempt_id}-breakdown"

    # Actually create git worktree
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)

    # Check if branch exists in git
    branch_check = subprocess.run(
        ['git', 'rev-parse', '--verify', branch_name],
        cwd=repo_path,
        capture_output=True
    )

    if branch_check.returncode == 0:
        # Branch exists, checkout in worktree
        subprocess.run(
            ['git', 'worktree', 'add', worktree_path, branch_name],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
    else:
        # Branch doesn't exist, create it
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
                               depth=0, max_depth=5, max_tools=50):
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
            max_tools=max_tools
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
            max_tools=max_tools
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
# Database Helper Functions
# ============================================================================

def create_session_record(db, session_id, description, repo_path, base_branch='main'):
    """Create the session record (single row)."""
    db.execute(
        """
        INSERT INTO session (id, session_id, description, repo_path, base_branch)
        VALUES (1, ?, ?, ?, ?)
        """,
        (session_id, description, repo_path, base_branch)
    )
    db.commit()


def get_session_record(db):
    """Get the session record."""
    cursor = db.execute("SELECT * FROM session WHERE id = 1")
    return cursor.fetchone()


def update_session_status(db, status):
    """Update session status."""
    db.execute("UPDATE session SET status = ? WHERE id = 1", (status,))
    if status == 'completed' or status == 'failed':
        db.execute("UPDATE session SET completed_at = CURRENT_TIMESTAMP WHERE id = 1")
    db.commit()


def create_goal(db, description, parent_id=None, goal_type='implementation',
                created_by_goal_id=None, git_branch=None, validation_commands=None):
    """
    Create a new goal and return its ID.

    Args:
        validation_commands: JSON string of validation commands
                            e.g., '[{"command": "bazel build //...", "expect": "success"}]'
    """
    cursor = db.execute(
        """
        INSERT INTO goals (description, parent_id, goal_type, created_by_goal_id, git_branch, validation_commands)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (description, parent_id, goal_type, created_by_goal_id, git_branch, validation_commands)
    )
    db.commit()
    return cursor.lastrowid


def get_goal(db, goal_id):
    """Get a goal by ID."""
    cursor = db.execute("SELECT * FROM goals WHERE id = ?", (goal_id,))
    return cursor.fetchone()


def get_child_goals(db, parent_id):
    """Get all child goals of a parent goal."""
    cursor = db.execute(
        "SELECT * FROM goals WHERE parent_id = ? ORDER BY id",
        (parent_id,)
    )
    return cursor.fetchall()


def get_root_goal(db):
    """Get the root goal (goal with no parent)."""
    cursor = db.execute("SELECT * FROM goals WHERE parent_id IS NULL LIMIT 1")
    return cursor.fetchone()


def update_goal_status(db, goal_id, status, git_branch=None):
    """Update goal status and optionally git branch."""
    if git_branch:
        db.execute(
            "UPDATE goals SET status = ?, git_branch = ? WHERE id = ?",
            (status, git_branch, goal_id)
        )
    else:
        db.execute(
            "UPDATE goals SET status = ? WHERE id = ?",
            (status, goal_id)
        )

    if status == 'completed' or status == 'failed':
        db.execute(
            "UPDATE goals SET completed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (goal_id,)
        )

    db.commit()


def create_attempt(db, goal_id, model_id=None, git_branch=None, worktree_path=None,
                   container_id=None, git_commit_sha=None):
    """Create a new attempt for a goal and return its ID."""
    cursor = db.execute(
        """
        INSERT INTO attempts
        (goal_id, model_id, git_branch, worktree_path, container_id, git_commit_sha)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (goal_id, model_id, git_branch, worktree_path, container_id, git_commit_sha)
    )
    db.commit()
    return cursor.lastrowid


def get_attempts(db, goal_id):
    """Get all attempts for a goal."""
    cursor = db.execute(
        "SELECT * FROM attempts WHERE goal_id = ? ORDER BY started_at",
        (goal_id,)
    )
    return cursor.fetchall()


def update_attempt_outcome(db, attempt_id, outcome, failure_reason=None):
    """Update attempt outcome."""
    db.execute(
        """
        UPDATE attempts
        SET outcome = ?, failure_reason = ?, completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (outcome, failure_reason, attempt_id)
    )
    db.commit()


def record_action(db, attempt_id, tool_name, parameters, result):
    """Record a tool action during an attempt."""
    cursor = db.execute(
        """
        INSERT INTO actions (attempt_id, tool_name, parameters, result)
        VALUES (?, ?, ?, ?)
        """,
        (attempt_id, tool_name, json.dumps(parameters), result)
    )
    db.commit()
    return cursor.lastrowid


def update_attempt_metadata(db, attempt_id, git_branch=None,
                            worktree_path=None, container_id=None,
                            git_commit_sha=None, final_commit_sha=None):
    """Update mutable metadata for an attempt."""
    fields = []
    values = []

    if git_branch is not None:
        fields.append("git_branch = ?")
        values.append(git_branch)
    if worktree_path is not None:
        fields.append("worktree_path = ?")
        values.append(worktree_path)
    if container_id is not None:
        fields.append("container_id = ?")
        values.append(container_id)
    if git_commit_sha is not None:
        fields.append("git_commit_sha = ?")
        values.append(git_commit_sha)
    if final_commit_sha is not None:
        fields.append("final_commit_sha = ?")
        values.append(final_commit_sha)

    if not fields:
        return

    values.append(attempt_id)
    query = f"UPDATE attempts SET {', '.join(fields)} WHERE id = ?"
    db.execute(query, tuple(values))
    db.commit()


def get_actions(db, attempt_id):
    """Get all actions for an attempt."""
    cursor = db.execute(
        "SELECT * FROM actions WHERE attempt_id = ? ORDER BY executed_at",
        (attempt_id,)
    )
    return cursor.fetchall()


# ============================================================================
# Model Utilities
# ============================================================================

def shorten_model_name(model_id):
    """
    Shorten model ID for use in branch names.

    Examples:
        'openrouter/anthropic/claude-haiku-4.5' -> 'haiku45'
        'openrouter/x-ai/grok-code-fast-1' -> 'grok-code-fast-1'
        'openrouter/google/gemini-2.5-flash' -> 'gemini25-flash'
    """
    # Remove 'openrouter/' prefix if present
    name = model_id.replace('openrouter/', '')

    # Common simplifications
    name = name.replace('anthropic/', '')
    name = name.replace('x-ai/', '')
    name = name.replace('google/', '')
    name = name.replace('openai/', '')
    name = name.replace('z-ai/', '')
    name = name.replace('qwen/', '')
    name = name.replace('claude-', '')

    # Remove dots and simplify version numbers
    name = name.replace('.', '')
    name = name.replace('_', '-')

    # Truncate if too long (git branch names should be reasonable)
    if len(name) > 20:
        name = name[:20]

    return name


# ============================================================================
# Git Management
# ============================================================================

class GitManager:
    """Manages git operations for goals and attempts using worktrees"""

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.worktrees_dir = self.repo_path / "worktrees"
        self.worktrees_dir.mkdir(exist_ok=True)

    def create_session_branch(self, session_id, base_branch='main'):
        """
        Create a session branch from base branch.
        Returns: branch_name
        """
        branch_name = f"session-{session_id}"

        # Create and checkout session branch
        subprocess.run(
            ['git', 'checkout', '-b', branch_name, base_branch],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        return branch_name

    def create_goal_branch(self, goal_id, parent_branch):
        """
        Create a goal branch from parent branch.
        Returns: branch_name
        """
        branch_name = f"goal-{goal_id}"

        subprocess.run(
            ['git', 'checkout', parent_branch],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        subprocess.run(
            ['git', 'checkout', '-b', branch_name],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        return branch_name

    def create_worktree_for_attempt(self, goal_id, attempt_id, base_branch, session_id=None, model_id=None):
        """
        Create a worktree with a new branch for an attempt.
        Returns: (worktree_path, branch_name, commit_sha)
        """
        # Include session_id and model for uniqueness across runs
        if session_id:
            if model_id:
                model_short = shorten_model_name(model_id)
                branch_name = f"s{session_id}-g{goal_id}-a{attempt_id}-m{model_short}"
            else:
                branch_name = f"s{session_id}-g{goal_id}-a{attempt_id}"
        else:
            branch_name = f"goal-{goal_id}-attempt-{attempt_id}"

        worktree_path = self.worktrees_dir / branch_name

        # Create worktree with new branch based on base_branch
        subprocess.run(
            ['git', 'worktree', 'add', str(worktree_path), '-b', branch_name, base_branch],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        # Get the commit SHA
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True
        )
        commit_sha = result.stdout.strip()

        return str(worktree_path), branch_name, commit_sha

    def get_current_commit_sha(self, path=None):
        """
        Get the current commit SHA at path (or repo root if not specified).
        Returns: commit_sha
        """
        target_path = path if path else self.repo_path

        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=target_path,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    def merge_branch(self, source_branch, target_branch, no_ff=True):
        """
        Merge source branch into target branch.
        """
        # Checkout target branch
        subprocess.run(
            ['git', 'checkout', target_branch],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        # Merge with --no-ff to preserve history
        merge_cmd = ['git', 'merge', source_branch]
        if no_ff:
            merge_cmd.insert(2, '--no-ff')

        subprocess.run(
            merge_cmd,
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

    def remove_worktree(self, worktree_path):
        """
        Remove a worktree directory.
        """
        subprocess.run(
            ['git', 'worktree', 'remove', str(worktree_path)],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

    def list_worktrees(self):
        """
        List all worktrees.
        Returns: list of worktree info dicts
        """
        result = subprocess.run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True
        )

        worktrees = []
        current_worktree = {}

        for line in result.stdout.split('\n'):
            if not line:
                if current_worktree:
                    worktrees.append(current_worktree)
                    current_worktree = {}
                continue

            if line.startswith('worktree '):
                current_worktree['path'] = line.split(' ', 1)[1]
            elif line.startswith('branch '):
                current_worktree['branch'] = line.split(' ', 1)[1]
            elif line.startswith('HEAD '):
                current_worktree['commit'] = line.split(' ', 1)[1]

        # Append the last worktree if there's no trailing blank line
        if current_worktree:
            worktrees.append(current_worktree)

        return worktrees

    def commit_worktree_changes(self, worktree_path, message):
        """
        Commit all changes in a worktree.

        Args:
            worktree_path: Path to the worktree
            message: Commit message

        Returns:
            commit_sha of the new commit, or None if nothing to commit
        """
        # Add all changes
        subprocess.run(
            ['git', 'add', '-A'],
            cwd=worktree_path,
            check=True,
            capture_output=True
        )

        # Check if there are changes to commit
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True
        )

        if not status_result.stdout.strip():
            # No changes to commit
            logging.debug(f"No changes to commit in {worktree_path}")
            return None

        # Commit the changes
        subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=worktree_path,
            check=True,
            capture_output=True
        )

        # Get the new commit SHA
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True
        )

        return result.stdout.strip()


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
        write_file,
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


def work_on_goal(db, goal_id, repo_path, image="shots-on-goal:latest", runtime="container",
                 model_id="openrouter/anthropic/claude-haiku-4.5", system_prompt=None, max_tools=50,
                 goal_working_branch=None, previous_attempts=None, merge_target_branch=None):
    """
    Work on a goal using an LLM agent with tool access.

    Args:
        db: Database connection
        goal_id: Goal ID to work on
        repo_path: Path to the repository
        image: Container image to use
        runtime: Container runtime ('container' or 'docker')
        model_id: LLM model to use (default: openrouter/anthropic/claude-haiku-4.5)
        system_prompt: Optional system prompt for the LLM
        max_tools: Maximum number of tool calls allowed (default: 50)
        goal_working_branch: Branch to base the attempt on (if None, uses session base_branch)
        previous_attempts: List of previous attempt results (for feedback loop)
        merge_target_branch: Branch to merge into on success (if None, uses goal_working_branch)

    Returns:
        dict with 'success', 'attempt_id', 'actions', 'outcome', 'response_text', 'attempt_branch'
    """
    # Get session info
    session_record = get_session_record(db)
    session_id = session_record['session_id']

    # Determine base branch for this attempt
    if goal_working_branch:
        base_branch = goal_working_branch
    else:
        base_branch = session_record['base_branch']

    # Initialize managers (use absolute path)
    abs_repo_path = os.path.abspath(repo_path)
    git_manager = GitManager(abs_repo_path)

    # Create attempt record first so we can derive branch/worktree names
    attempt_id = create_attempt(db, goal_id=goal_id, model_id=model_id)

    # Initialize variables for exception handling
    actions = []
    pending_actions_for_db = []
    actions_flushed = False
    last_activity_time = [time.time()]  # Use list for mutability in closure

    def flush_pending_actions():
        nonlocal actions_flushed
        if actions_flushed:
            return
        for pending in pending_actions_for_db:
            record_action(
                db,
                attempt_id,
                pending['tool'],
                pending['arguments'],
                pending['output']
            )
        actions_flushed = True
        pending_actions_for_db.clear()

    container = ContainerManager(image=image, runtime=runtime)

    try:
        # Create worktree for the attempt using the real attempt ID
        worktree_path, branch_name, commit_sha = git_manager.create_worktree_for_attempt(
            goal_id,
            attempt_id=attempt_id,
            base_branch=base_branch,
            session_id=session_id,
            model_id=model_id
        )

        # Persist metadata we already know
        update_attempt_metadata(
            db,
            attempt_id,
            git_branch=branch_name,
            worktree_path=worktree_path,
            git_commit_sha=commit_sha
        )

        logging.info(f"[Attempt {attempt_id}] Model: {model_id}")
        logging.info(f"[Attempt {attempt_id}] Branch: {branch_name}")
        logging.info(f"[Attempt {attempt_id}] Worktree: {worktree_path}")

        container_id = container.start(worktree_path)

        # Record container ID once the container is running
        update_attempt_metadata(db, attempt_id, container_id=container_id)

        # Initialize tool executor
        tools_executor = ToolExecutor(container)

        # Create tool functions for LLM
        tool_functions = create_tool_functions(tools_executor)

        # Get the goal description
        goal = get_goal(db, goal_id)
        goal_description = goal['description']

        logging.info(f"[Attempt {attempt_id}] [{model_id}] Working on goal: {goal_description}")

        # Set up after_call hook to record tool calls
        def after_call(tool, tool_call, tool_result):
            """Record each tool call in the database"""
            # Check for timeout (no activity for 2 minutes)
            elapsed = time.time() - last_activity_time[0]
            if elapsed > 120:
                logging.error(f"[Attempt {attempt_id}] Timeout: No activity for {elapsed:.1f} seconds")
                raise LLMTimeoutError(f"No API activity for {elapsed:.1f} seconds (timeout: 120s)")

            # Update activity timestamp
            last_activity_time[0] = time.time()

            # Get tool name safely - handles both llm.Tool instances and plain functions
            tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))

            # Log tool call with sanitized arguments
            args_str = str(tool_call.arguments)[:100]  # Truncate long args
            logging.info(f"  Tool {len(actions)+1}/{max_tools}: {tool_name}({args_str})")

            pending_actions_for_db.append({
                "tool": tool_name,
                "arguments": tool_call.arguments,
                "output": tool_result.output
            })
            actions.append({
                "tool": tool_name,
                "arguments": tool_call.arguments,
                "output": tool_result.output
            })

            # Check if we've exceeded the tool limit
            if len(actions) >= max_tools:
                logging.warning(f"[Attempt {attempt_id}] Tool limit ({max_tools}) reached")
                raise ToolLimitExceeded(f"Tool limit of {max_tools} exceeded")

        # Get LLM model
        model = llm.get_model(model_id)

        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = f"""You are an autonomous coding agent working on a specific goal in a git repository.

You have access to tools to read files, search code, modify files, and run Bazel commands.

**Important constraints:**
- You have {max_tools} tool calls to achieve this goal
- Use tools efficiently to stay within the limit
- For Bazel projects: use MODULE.bazel with bzlmod, NOT WORKSPACE files
- The container has bzlmod enabled by default (--enable_bzlmod, --noenable_workspace)

Your task is to work towards achieving the goal. You should:
1. Explore the repository to understand its structure
2. Make necessary changes to achieve the goal
3. Test your changes using Bazel build/test commands when appropriate
4. Be methodical and explain your reasoning

When you have successfully achieved the goal (or determined it cannot be achieved), explain your final status clearly."""

            # Add feedback from previous attempts if any
            if previous_attempts:
                logging.info(f"[Attempt {attempt_id}] Including feedback from {len(previous_attempts)} previous attempt(s)")

                feedback_section = "\n\n**IMPORTANT - Previous Attempts:**\n"
                feedback_section += f"This goal has been attempted {len(previous_attempts)} time(s) before. Learn from these attempts:\n\n"

                for i, prev in enumerate(previous_attempts, 1):
                    outcome = prev['outcome']
                    feedback_section += f"Attempt {i}: {outcome}\n"

                    # Log summary of each previous attempt
                    logging.info(f"[Attempt {attempt_id}]   Previous attempt {i}: {outcome}")

                    # Include error details if available
                    if 'error' in prev:
                        error_msg = str(prev['error'])[:200]  # Truncate long errors
                        feedback_section += f"  Error: {error_msg}\n"
                        logging.info(f"[Attempt {attempt_id}]     Error: {error_msg[:100]}")

                    # Include validation failures if available
                    if outcome == 'error' and 'actions' in prev and len(prev['actions']) > 0:
                        # The validation failure info would be in the final actions/result
                        feedback_section += f"  ({len(prev['actions'])} tool calls were made)\n"

                feedback_section += "\n**What to do differently:**\n"
                feedback_section += "- Analyze what went wrong in previous attempts\n"
                feedback_section += "- Try a different approach or fix the specific errors\n"
                feedback_section += "- Pay attention to validation failures from previous attempts\n"

                system_prompt += feedback_section

        # Execute LLM chain with tools
        logging.info(f"[Attempt {attempt_id}] Starting LLM agent with model {model_id}")
        chain = model.chain(
            goal_description,
            system=system_prompt,
            tools=tool_functions,
            after_call=after_call
        )

        # Get the response text (timeout is checked in after_call)
        response_text = chain.text()

        logging.info(f"[Attempt {attempt_id}] LLM agent completed - {len(actions)} tools used")
        logging.debug(f"[Attempt {attempt_id}] Response: {response_text[:200]}...")

        # ============================================================
        # Determine outcome
        # ============================================================

        flush_pending_actions()

        # Check if code changes were made
        git_status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True
        )
        has_changes = bool(git_status_result.stdout.strip())

        # Get validation commands if present
        validation_commands_json = goal['validation_commands']
        has_validation = bool(validation_commands_json)

        if has_validation:
            # Goal has validation commands - must have changes AND pass validation
            if not has_changes:
                logging.info(f"[Attempt {attempt_id}] No code changes made (but validation required)")
                outcome = "error"
                failure_reason = "No code changes made"
            else:
                logging.info(f"[Attempt {attempt_id}] Code changes detected")

                # Run validation commands
                validation_passed = True
                validation_errors = []

                try:
                    validation_commands = json.loads(validation_commands_json)
                    logging.info(f"[Attempt {attempt_id}] Running {len(validation_commands)} validation commands")

                    for i, val_cmd in enumerate(validation_commands, 1):
                        command = val_cmd.get('command', '')
                        expect = val_cmd.get('expect', 'success')

                        logging.info(f"[Attempt {attempt_id}]   Validation {i}/{len(validation_commands)}: {command}")

                        # Run the validation command in container
                        result = container.exec(command, timeout=120)

                        # Check if it succeeded (expect == 'success' means exit code 0)
                        if expect == 'success':
                            if result.returncode == 0:
                                logging.info(f"[Attempt {attempt_id}]   ✓ Validation {i} passed")
                            else:
                                logging.warning(f"[Attempt {attempt_id}]   ✗ Validation {i} failed (exit code {result.returncode})")
                                validation_passed = False
                                validation_errors.append(f"Command '{command}' failed with exit code {result.returncode}")
                                # Log stderr for debugging
                                if result.stderr:
                                    logging.debug(f"[Attempt {attempt_id}]     stderr: {result.stderr[:200]}")
                        else:
                            # For other expectations, we could extend this in the future
                            logging.warning(f"[Attempt {attempt_id}]   ? Unknown expectation: {expect}")

                except json.JSONDecodeError as e:
                    logging.warning(f"[Attempt {attempt_id}] Failed to parse validation_commands JSON: {e}")
                    validation_passed = False
                    validation_errors.append(f"JSON parsing error: {e}")

                # Determine final outcome
                if validation_passed:
                    outcome = "success"
                    failure_reason = None
                    logging.info(f"[Attempt {attempt_id}] ✓ All validations passed")
                else:
                    outcome = "error"
                    failure_reason = "Validation failed: " + "; ".join(validation_errors)
                    logging.info(f"[Attempt {attempt_id}] ✗ Validation failed")
        else:
            # No validation commands - success if LLM completed without error
            # This handles exploratory goals that don't require code changes
            outcome = "success"
            failure_reason = None
            if has_changes:
                logging.info(f"[Attempt {attempt_id}] ✓ Completed with code changes (no validation required)")
            else:
                logging.info(f"[Attempt {attempt_id}] ✓ Completed successfully (no validation required)")

        logging.info(f"[Attempt {attempt_id}] Completed with outcome: {outcome}")

        # Commit all changes made during the attempt
        final_commit_sha = git_manager.commit_worktree_changes(
            worktree_path,
            f"Attempt {attempt_id}: {outcome}\n\nGoal: {goal_description}"
        )
        if final_commit_sha:
            logging.info(f"[Attempt {attempt_id}] Committed changes: {final_commit_sha[:8]} on branch {branch_name}")
            logging.info(f"[Attempt {attempt_id}]   View changes: git show {final_commit_sha[:8]}")
            logging.info(f"[Attempt {attempt_id}]   Worktree files: {worktree_path}")
            update_attempt_metadata(db, attempt_id, final_commit_sha=final_commit_sha)
        else:
            logging.info(f"[Attempt {attempt_id}] No changes to commit")

        # If successful, merge attempt branch into goal's working branch
        # Use merge_target_branch if specified, otherwise use goal_working_branch
        target_branch = merge_target_branch if merge_target_branch else goal_working_branch
        if outcome == "success" and target_branch:
            logging.info(f"[Attempt {attempt_id}] Merging {branch_name} into {target_branch}")
            try:
                git_manager.merge_branch(branch_name, target_branch, no_ff=True)
                logging.info(f"[Attempt {attempt_id}] ✓ Merged successfully")
            except subprocess.CalledProcessError as e:
                logging.error(f"[Attempt {attempt_id}] Failed to merge: {e}")
                # Update outcome to error since merge failed
                outcome = "error"
                failure_reason = f"Merge failed: {e}"

        # Update attempt with outcome
        update_attempt_outcome(db, attempt_id, outcome, failure_reason)

        return {
            'success': outcome == "success",
            'attempt_id': attempt_id,
            'actions': actions,
            'outcome': outcome,
            'response_text': response_text,
            'attempt_branch': branch_name
        }

    except ToolLimitExceeded as e:
        # Tool limit exceeded - report this specific outcome
        logging.warning(f"[Attempt {attempt_id}] Tool limit exceeded")
        flush_pending_actions()

        # Commit changes even if we hit the limit
        if 'worktree_path' in locals():
            final_commit_sha = git_manager.commit_worktree_changes(
                worktree_path,
                f"Attempt {attempt_id}: tool_limit_exceeded\n\nGoal: {goal_description}"
            )
            if final_commit_sha:
                logging.info(f"[Attempt {attempt_id}] Committed changes: {final_commit_sha[:8]}")
                update_attempt_metadata(db, attempt_id, final_commit_sha=final_commit_sha)

        update_attempt_outcome(db, attempt_id, "tool_limit_exceeded", str(e))

        return {
            'success': False,
            'attempt_id': attempt_id,
            'actions': actions,
            'outcome': "tool_limit_exceeded",
            'error': str(e),
            'attempt_branch': branch_name if 'branch_name' in locals() else None
        }

    except LLMTimeoutError as e:
        # LLM call timed out - report this specific outcome
        logging.warning(f"[Attempt {attempt_id}] LLM timeout: {e}")
        flush_pending_actions()

        # Commit changes even on timeout
        if 'worktree_path' in locals():
            final_commit_sha = git_manager.commit_worktree_changes(
                worktree_path,
                f"Attempt {attempt_id}: llm_timeout\n\nGoal: {goal_description}"
            )
            if final_commit_sha:
                logging.info(f"[Attempt {attempt_id}] Committed changes: {final_commit_sha[:8]}")
                update_attempt_metadata(db, attempt_id, final_commit_sha=final_commit_sha)

        update_attempt_outcome(db, attempt_id, "llm_timeout", str(e))

        return {
            'success': False,
            'attempt_id': attempt_id,
            'actions': actions,
            'outcome': "llm_timeout",
            'error': str(e),
            'attempt_branch': branch_name if 'branch_name' in locals() else None
        }

    except Exception as e:
        # If something went wrong, record as error
        logging.error(f"[Attempt {attempt_id}] Error: {e}")
        flush_pending_actions()

        # Commit changes even on error
        if 'worktree_path' in locals():
            try:
                final_commit_sha = git_manager.commit_worktree_changes(
                    worktree_path,
                    f"Attempt {attempt_id}: error\n\nGoal: {goal_description}\n\nError: {str(e)}"
                )
                if final_commit_sha:
                    logging.info(f"[Attempt {attempt_id}] Committed changes: {final_commit_sha[:8]}")
                    update_attempt_metadata(db, attempt_id, final_commit_sha=final_commit_sha)
            except Exception as commit_error:
                logging.warning(f"[Attempt {attempt_id}] Failed to commit changes: {commit_error}")

        update_attempt_outcome(db, attempt_id, "error", str(e))

        return {
            'success': False,
            'attempt_id': attempt_id,
            'actions': actions,
            'outcome': "error",
            'error': str(e),
            'attempt_branch': branch_name if 'branch_name' in locals() else None
        }

    finally:
        flush_pending_actions()
        # Always clean up container
        container.stop()
        logging.debug(f"[Attempt {attempt_id}] Cleaned up container")


# ============================================================================
# Goal Decomposition
# ============================================================================

def decompose_goal(db, goal_id, attempt_results, model_id="openrouter/anthropic/claude-haiku-4.5"):
    """
    Decompose a goal into sub-goals using LLM.

    Args:
        db: Database connection
        goal_id: Goal ID that needs decomposition
        attempt_results: List of result dicts from work_on_goal() attempts
        model_id: LLM model to use

    Returns:
        List of sub-goal IDs created
    """
    goal = get_goal(db, goal_id)
    goal_description = goal['description']

    logging.info(f"[Goal {goal_id}] Decomposing goal after {len(attempt_results)} attempts")

    # Build context about all attempts
    attempts_summary = []
    for i, result in enumerate(attempt_results, 1):
        outcome = result['outcome']
        actions = result.get('actions', [])
        tools_used = [action['tool'] for action in actions]
        tool_summary = ", ".join(set(tools_used[:10]))  # First 10 unique tools

        attempt_info = f"Attempt {i}: {outcome}"
        if actions:
            attempt_info += f" ({len(actions)} tool calls: {tool_summary})"
        if 'error' in result:
            error_msg = str(result['error'])[:150]
            attempt_info += f" - Error: {error_msg}"

        attempts_summary.append(attempt_info)

    # Prompt for decomposition
    decomposition_prompt = f"""A goal could not be achieved after multiple attempts. Analyze the situation and break down the goal into smaller, concrete sub-goals.

**Original Goal:**
{goal_description}

**Attempts Made:**
{chr(10).join(f'- {summary}' for summary in attempts_summary)}

**Your task:**
Analyze what went wrong in the attempts above and break this goal down into 3-5 concrete, actionable sub-goals that:
1. AVOID the mistakes/errors from previous attempts
2. Are VERY SPECIFIC about what to create/modify (not exploratory or documentation tasks)
3. Are smaller in scope (achievable in <20 tool calls each)
4. Build toward achieving the original goal
5. Build on each other sequentially (each sub-goal uses the results of previous ones)
6. Have concrete validation commands that test functionality

**What makes a good sub-goal:**
✓ "Create MODULE.bazel with rules_python version 0.31.0"
✓ "Write BUILD file in src/ to build mylib.py as py_library"
✓ "Add py_binary target for main.py in root BUILD file"
✗ "Explore the codebase structure" (too vague)
✗ "Document the migration process" (not testable)
✗ "Create comprehensive BUILD files" (too broad)

**Important:**
- Use MODULE.bazel with bzlmod, NOT WORKSPACE files
- Each sub-goal MUST have validation commands that test the build/functionality
- Focus on making things buildable, not on documentation

**Output format:**
Return ONLY valid JSON (you can wrap in ```json if you want):
[
  {{
    "description": "Create MODULE.bazel with rules_python version 0.31.0 and declare mylib module",
    "validation": [
      {{"command": "test -f MODULE.bazel", "expect": "success"}},
      {{"command": "bazel query //...", "expect": "success"}}
    ]
  }},
  {{
    "description": "Write BUILD file in src/ to build mylib.py as py_library target",
    "validation": [
      {{"command": "bazel build //src:mylib", "expect": "success"}}
    ]
  }}
]"""

    # Get LLM to decompose
    model = llm.get_model(model_id)
    logging.debug(f"[Goal {goal_id}] Requesting decomposition from {model_id}")

    try:
        response = model.prompt(decomposition_prompt)
        response_text = response.text().strip()
    except Exception as e:
        logging.error(f"[Goal {goal_id}] Decomposition failed: {e}")
        # Mark goal as failed instead of creating retry loop
        update_goal_status(db, goal_id, 'failed')
        return []

    logging.debug(f"[Goal {goal_id}] Decomposition response: {response_text}")

    # Parse JSON response
    sub_goals = []
    try:
        # Try to extract fenced code block first
        extracted = extract_fenced_code_block(response_text, last=False)
        json_text = extracted if extracted else response_text

        sub_goals_data = json.loads(json_text)

        if not isinstance(sub_goals_data, list):
            raise ValueError("Expected JSON array")

        sub_goals = sub_goals_data

    except (json.JSONDecodeError, ValueError) as e:
        logging.warning(f"[Goal {goal_id}] Failed to parse JSON response: {e}")
        logging.warning(f"[Goal {goal_id}] Response was: {response_text[:200]}")
        # Fallback: create a single retry goal
        sub_goals = [{
            "description": f"Retry: {goal_description}",
            "validation": []
        }]

    if not sub_goals:
        logging.warning(f"[Goal {goal_id}] No sub-goals in response")
        sub_goals = [{
            "description": f"Retry: {goal_description}",
            "validation": []
        }]

    logging.info(f"[Goal {goal_id}] Created {len(sub_goals)} sub-goals")

    # Create sub-goal records in database
    sub_goal_ids = []
    for i, sub_goal_data in enumerate(sub_goals, 1):
        description = sub_goal_data.get('description', f'Sub-goal {i}')
        validation = sub_goal_data.get('validation', [])

        # Store validation commands as JSON
        validation_json = json.dumps(validation) if validation else None

        sub_goal_id = create_goal(
            db,
            description=description,
            parent_id=goal_id,
            goal_type='implementation',
            created_by_goal_id=goal_id,
            validation_commands=validation_json
        )
        sub_goal_ids.append(sub_goal_id)
        logging.info(f"  Sub-goal {i}/{len(sub_goals)} (ID: {sub_goal_id}): {description[:80]}")
        if validation:
            logging.info(f"    Validation: {len(validation)} commands")
            for j, cmd in enumerate(validation, 1):
                logging.info(f"      {j}. {cmd.get('command', 'N/A')}")

    # Mark parent goal as decomposed
    update_goal_status(db, goal_id, 'decomposed')

    return sub_goal_ids


# ============================================================================
# Work Orchestration
# ============================================================================

def work_on_goal_recursive(db, goal_id, repo_path, image, runtime, model_a, model_b,
                          max_tools=50, max_decompositions=5, depth=0, max_depth=5,
                          parent_working_branch=None, current_model=None,
                          decomposition_count=0):
    """
    Work on a goal recursively with ping-pong model alternation.

    Args:
        db: Database connection
        goal_id: Goal ID to work on
        repo_path: Path to repository
        image: Container image
        runtime: Container runtime
        model_a: First model ID
        model_b: Second model ID
        max_tools: Maximum tool calls per attempt
        max_decompositions: Maximum number of decompositions before giving up
        depth: Current recursion depth
        max_depth: Maximum recursion depth
        parent_working_branch: Parent goal's working branch (None for root goal)
        current_model: Which model should attempt this goal (None = start with A)
        decomposition_count: How many decompositions have occurred so far

    Returns:
        bool: True if goal completed successfully
    """
    indent = "  " * depth
    goal = get_goal(db, goal_id)
    goal_desc = goal['description'][:80]

    logging.info(f"{indent}[Goal {goal_id}] Working on: {goal_desc}")

    # Check decomposition limit
    if decomposition_count >= max_decompositions:
        logging.warning(f"{indent}[Goal {goal_id}] Max decompositions ({max_decompositions}) reached - marking as failed")
        update_goal_status(db, goal_id, 'failed')
        return False

    # Check depth limit
    if depth >= max_depth:
        logging.warning(f"{indent}[Goal {goal_id}] Max depth ({max_depth}) reached - marking as failed")
        update_goal_status(db, goal_id, 'failed')
        return False

    # Create or get goal's working branch
    git_manager = GitManager(os.path.abspath(repo_path))
    session_record = get_session_record(db)
    session_id = session_record['session_id']

    if goal['git_branch']:
        # Use existing branch
        goal_working_branch = goal['git_branch']
        logging.debug(f"{indent}[Goal {goal_id}] Using existing branch: {goal_working_branch}")
    else:
        # Create new branch for this goal
        if parent_working_branch:
            base = parent_working_branch
        else:
            # Root goal - use session base branch
            base = session_record['base_branch']

        # Include session ID to avoid conflicts across runs
        goal_working_branch = f"s{session_id}-goal-{goal_id}"
        logging.info(f"{indent}[Goal {goal_id}] Creating working branch: {goal_working_branch} from {base}")

        # Create the branch
        subprocess.run(
            ['git', 'checkout', base],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ['git', 'checkout', '-b', goal_working_branch],
            cwd=repo_path,
            check=True,
            capture_output=True
        )

        # Store branch in database
        update_goal_status(db, goal_id, 'in_progress', git_branch=goal_working_branch)

    # Mark goal as in progress (if not already done above)
    if goal['status'] != 'in_progress':
        update_goal_status(db, goal_id, 'in_progress')

    # Determine which model to use (default to A if not specified)
    if current_model is None:
        model_to_use = model_a
        other_model = model_b
    else:
        model_to_use = current_model
        other_model = model_b if current_model == model_a else model_a

    logging.info(f"{indent}[Goal {goal_id}] {shorten_model_name(model_to_use)} attempting...")

    # Single attempt with current model
    start_time = time.time()
    result = work_on_goal(
        db=db,
        goal_id=goal_id,
        repo_path=repo_path,
        image=image,
        runtime=runtime,
        model_id=model_to_use,
        max_tools=max_tools,
        goal_working_branch=goal_working_branch,
        merge_target_branch=goal_working_branch
    )
    elapsed = time.time() - start_time

    # Handle the outcome
    if result['outcome'] == 'success':
        logging.info(f"{indent}[Goal {goal_id}] ✓ {shorten_model_name(model_to_use)} succeeded in {elapsed:.1f}s!")
        update_goal_status(db, goal_id, 'completed')
        return True
    else:
        logging.info(f"{indent}[Goal {goal_id}] ✗ {shorten_model_name(model_to_use)} failed: {result['outcome']} (took {elapsed:.1f}s)")

    # Failed - decompose with other model
    logging.info(f"{indent}[Goal {goal_id}] Flipping to {shorten_model_name(other_model)} for decomposition...")

    sub_goal_ids = decompose_goal(db, goal_id, [result], model_id=other_model)

    if not sub_goal_ids:
        logging.warning(f"{indent}[Goal {goal_id}] No sub-goals created - marking as failed")
        update_goal_status(db, goal_id, 'failed')
        return False

    # Work on each sub-goal sequentially with the decomposing model
    # STOP on first failure (fail-fast)
    all_succeeded = True

    for i, sub_goal_id in enumerate(sub_goal_ids, 1):
        logging.info(f"{indent}[Goal {goal_id}] Working on sub-goal {i}/{len(sub_goal_ids)}")

        success = work_on_goal_recursive(
            db=db,
            goal_id=sub_goal_id,
            repo_path=repo_path,
            image=image,
            runtime=runtime,
            model_a=model_a,
            model_b=model_b,
            max_tools=max_tools,
            max_decompositions=max_decompositions,
            depth=depth + 1,
            max_depth=max_depth,
            parent_working_branch=goal_working_branch,  # Sub-goals build on parent's branch
            current_model=other_model,  # The model that decomposed works on sub-goals
            decomposition_count=decomposition_count + 1  # Increment decomposition count
        )

        if not success:
            all_succeeded = False
            logging.warning(f"{indent}[Goal {goal_id}] Sub-goal {sub_goal_id} failed - stopping remaining sub-goals")
            break  # Fail fast - don't process remaining sub-goals
        else:
            logging.info(f"{indent}[Goal {goal_id}] Sub-goal {sub_goal_id} succeeded - changes merged into {goal_working_branch}")

    # Update parent goal status based on sub-goals
    if all_succeeded:
        logging.info(f"{indent}[Goal {goal_id}] ✓ All sub-goals completed - marking as completed")
        update_goal_status(db, goal_id, 'completed')
        return True
    else:
        logging.warning(f"{indent}[Goal {goal_id}] ✗ Some sub-goals failed - marking as failed")
        update_goal_status(db, goal_id, 'failed')
        return False


# ============================================================================
# Session Management
# ============================================================================

def get_sessions_dir():
    """Get or create the sessions directory."""
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir


def create_session(goal_description, repo_path, base_branch='main'):
    """
    Create a new session with database and directory.
    Returns (session_dir, db).
    """
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = get_sessions_dir() / f"session-{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    db_path = session_dir / "goals.db"
    db = init_database(str(db_path))

    # Create session record
    create_session_record(db, session_id, goal_description,
                         os.path.abspath(repo_path), base_branch)

    return session_dir, db


def load_session(session_path):
    """
    Load an existing session.
    Returns (session_dir, db, session_record).
    """
    session_dir = Path(session_path)
    if not session_dir.exists():
        logging.error(f"Session directory not found: {session_dir}")
        sys.exit(1)

    db_path = session_dir / "goals.db"
    if not db_path.exists():
        logging.error(f"Database not found: {db_path}")
        sys.exit(1)

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    # Enable foreign key constraints
    db.execute("PRAGMA foreign_keys = ON")

    session_record = get_session_record(db)

    logging.info(f"Loaded session: {session_dir}")
    return session_dir, db, session_record


def list_sessions():
    """List all sessions."""
    sessions_dir = get_sessions_dir()
    sessions = sorted(sessions_dir.glob("session-*"))

    if not sessions:
        print("No sessions found.")
        return

    print("\nSessions:")
    print("-" * 80)
    for session_dir in sessions:
        db_path = session_dir / "goals.db"
        if not db_path.exists():
            continue

        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")

        session_record = get_session_record(db)
        if session_record:
            status = session_record['status']
            description = session_record['description']
            started_at = session_record['started_at']

            print(f"{session_dir.name:40s} [{status:10s}] {description}")
            print(f"  Started: {started_at}")

        db.close()
    print("-" * 80)


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

    # Session management
    parser.add_argument(
        '--resume',
        metavar='SESSION',
        help='Resume an existing session'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all sessions'
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
        '--use-v2',
        action='store_true',
        help='Use V2 schema and orchestration (experimental)'
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

    # Handle list command
    if args.list:
        list_sessions()
        return

    # Handle resume command
    if args.resume:
        session_dir, db, session_record = load_session(args.resume)
        logging.info(f"Resuming: {session_record['description']}")
        logging.info(f"Repo: {session_record['repo_path']}")
        # TODO: Resume work on goals
        db.close()
        return

    # Handle new session
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
    # V2 Execution Path
    # ========================================================================
    if args.use_v2:
        logging.info("=" * 80)
        logging.info("Using V2 schema and orchestration")
        logging.info("=" * 80)

        # Create V2 database
        db_path = f"shots-on-goal-v2-{int(time.time())}.db"
        db = init_database_v2(db_path)
        logging.info(f"Created V2 database: {db_path}")

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
            ('write_file', 'Write content to a file'),
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
                max_tools=args.max_tools
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

    # ========================================================================
    # V1 Execution Path (original)
    # ========================================================================

    # Create new session
    session_dir, db = create_session(args.goal, str(repo_path), base_branch)
    logging.info(f"Created session: {session_dir.name}")

    # Create root goal
    root_goal_id = create_goal(db, args.goal)
    logging.info(f"Created root goal (ID: {root_goal_id}): {args.goal}")

    # Start working on the root goal (with decomposition)
    logging.info("=" * 80)
    logging.info("Starting work on goal (with recursive decomposition)...")
    logging.info("=" * 80)

    try:
        success = work_on_goal_recursive(
            db=db,
            goal_id=root_goal_id,
            repo_path=str(repo_path),
            image=args.image,
            runtime=detect_container_runtime(),
            model_a=args.model_a,
            model_b=args.model_b,
            max_tools=args.max_tools,
            max_decompositions=args.max_goal_breakdowns
        )

        logging.info("=" * 80)
        logging.info("Goal tree completed!")
        logging.info("=" * 80)

        # Get final status
        final_goal = get_goal(db, root_goal_id)
        logging.info(f"Root goal status: {final_goal['status']}")

        if success:
            logging.info("✓ Root goal completed successfully!")
        else:
            logging.info("✗ Root goal did not complete successfully")

        # Show goal tree summary
        logging.info("")
        logging.info("--- Goal Tree Summary ---")
        all_goals = db.execute("SELECT id, description, status, parent_id FROM goals ORDER BY id").fetchall()
        for goal in all_goals:
            depth = 0
            parent_id = goal['parent_id']
            while parent_id:
                depth += 1
                parent = get_goal(db, parent_id)
                parent_id = parent['parent_id'] if parent else None

            indent = "  " * depth
            status_icon = "✓" if goal['status'] == 'completed' else ("✗" if goal['status'] == 'failed' else "•")
            logging.info(f"{indent}{status_icon} Goal {goal['id']}: {goal['description'][:60]} [{goal['status']}]")

        # Show worktree/branch information
        logging.info("")
        logging.info("--- Worktrees & Branches ---")
        all_attempts = db.execute("""
            SELECT a.id, a.goal_id, a.model_id, a.git_branch, a.worktree_path, a.outcome, a.final_commit_sha
            FROM attempts a
            ORDER BY a.id
        """).fetchall()

        for attempt in all_attempts:
            if attempt['git_branch'] and attempt['worktree_path']:
                status_icon = "✓" if attempt['outcome'] == 'success' else "✗"
                model_short = shorten_model_name(attempt['model_id']) if attempt['model_id'] else 'unknown'
                commit_info = f" (commit: {attempt['final_commit_sha'][:8]})" if attempt['final_commit_sha'] else ""
                logging.info(f"{status_icon} Attempt {attempt['id']} (Goal {attempt['goal_id']}, Model: {model_short}): {attempt['outcome']}{commit_info}")
                logging.info(f"    Branch: {attempt['git_branch']}")
                logging.info(f"    Worktree: {attempt['worktree_path']}")
                if attempt['final_commit_sha']:
                    logging.info(f"    View: git show {attempt['final_commit_sha'][:8]}")

    except KeyboardInterrupt:
        logging.info("")
        logging.info("Interrupted by user. Session saved.")
        update_session_status(db, 'interrupted')
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        update_session_status(db, 'failed')
        raise
    finally:
        db.close()

    logging.info("")
    logging.info(f"Session saved to: {session_dir}")


if __name__ == "__main__":
    main()

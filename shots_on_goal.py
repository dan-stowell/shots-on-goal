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
import llm


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
            FOREIGN KEY (parent_id) REFERENCES goals(id),
            FOREIGN KEY (created_by_goal_id) REFERENCES goals(id)
        )
    """)

    # Attempts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id INTEGER NOT NULL,
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

    # Add final_commit_sha column if it doesn't exist (for existing databases)
    cursor.execute("PRAGMA table_info(attempts)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'final_commit_sha' not in columns:
        cursor.execute("ALTER TABLE attempts ADD COLUMN final_commit_sha TEXT")

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
                created_by_goal_id=None, git_branch=None):
    """
    Create a new goal and return its ID.
    """
    cursor = db.execute(
        """
        INSERT INTO goals (description, parent_id, goal_type, created_by_goal_id, git_branch)
        VALUES (?, ?, ?, ?, ?)
        """,
        (description, parent_id, goal_type, created_by_goal_id, git_branch)
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


def create_attempt(db, goal_id, git_branch=None, worktree_path=None,
                   container_id=None, git_commit_sha=None):
    """Create a new attempt for a goal and return its ID."""
    cursor = db.execute(
        """
        INSERT INTO attempts
        (goal_id, git_branch, worktree_path, container_id, git_commit_sha)
        VALUES (?, ?, ?, ?, ?)
        """,
        (goal_id, git_branch, worktree_path, container_id, git_commit_sha)
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

    def create_worktree_for_attempt(self, goal_id, attempt_id, base_branch, session_id=None):
        """
        Create a worktree with a new branch for an attempt.
        Returns: (worktree_path, branch_name, commit_sha)
        """
        # Include session_id for uniqueness across runs
        if session_id:
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
                 model_id="openrouter/anthropic/claude-haiku-4.5", system_prompt=None, max_tools=20):
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
        max_tools: Maximum number of tool calls allowed (default: 20)

    Returns:
        dict with 'success', 'attempt_id', 'actions', 'outcome', 'response_text'
    """
    # Get session info for base branch
    session_record = get_session_record(db)
    base_branch = session_record['base_branch']
    session_id = session_record['session_id']

    # Initialize managers (use absolute path)
    abs_repo_path = os.path.abspath(repo_path)
    git_manager = GitManager(abs_repo_path)

    # Create attempt record first so we can derive branch/worktree names
    attempt_id = create_attempt(db, goal_id=goal_id)

    # Initialize variables for exception handling
    actions = []
    container = ContainerManager(image=image, runtime=runtime)

    try:
        # Create worktree for the attempt using the real attempt ID
        worktree_path, branch_name, commit_sha = git_manager.create_worktree_for_attempt(
            goal_id,
            attempt_id=attempt_id,
            base_branch=base_branch,
            session_id=session_id
        )

        # Persist metadata we already know
        update_attempt_metadata(
            db,
            attempt_id,
            git_branch=branch_name,
            worktree_path=worktree_path,
            git_commit_sha=commit_sha
        )

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

        logging.info(f"[Attempt {attempt_id}] Working on goal: {goal_description}")

        # Set up after_call hook to record tool calls
        def after_call(tool, tool_call, tool_result):
            """Record each tool call in the database"""
            # Get tool name safely - handles both llm.Tool instances and plain functions
            tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))

            # Log tool call with sanitized arguments
            args_str = str(tool_call.arguments)[:100]  # Truncate long args
            logging.info(f"  Tool {len(actions)+1}/{max_tools}: {tool_name}({args_str})")

            record_action(
                db,
                attempt_id,
                tool_name,
                tool_call.arguments,
                tool_result.output
            )
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
            system_prompt = """You are an autonomous coding agent working on a specific goal in a git repository.

You have access to tools to read files, search code, modify files, and run Bazel commands.

Your task is to work towards achieving the goal. You should:
1. Explore the repository to understand its structure
2. Make necessary changes to achieve the goal
3. Test your changes using Bazel build/test commands when appropriate
4. Be methodical and explain your reasoning

When you have successfully achieved the goal (or determined it cannot be achieved), explain your final status clearly."""

        # Execute LLM chain with tools
        logging.info(f"[Attempt {attempt_id}] Starting LLM agent with model {model_id}")
        chain = model.chain(
            goal_description,
            system=system_prompt,
            tools=tool_functions,
            after_call=after_call
        )

        # Get the response text
        response_text = chain.text()

        logging.info(f"[Attempt {attempt_id}] LLM agent completed - {len(actions)} tools used")
        logging.debug(f"[Attempt {attempt_id}] Response: {response_text[:200]}...")

        # ============================================================
        # Determine outcome
        # ============================================================

        # For now, we mark it as successful if the LLM completed without errors
        # In the future, we can parse the response to determine success/failure
        outcome = "success"
        failure_reason = None

        logging.info(f"[Attempt {attempt_id}] Completed with outcome: {outcome}")

        # Commit all changes made during the attempt
        final_commit_sha = git_manager.commit_worktree_changes(
            worktree_path,
            f"Attempt {attempt_id}: {outcome}\n\nGoal: {goal_description}"
        )
        if final_commit_sha:
            logging.info(f"[Attempt {attempt_id}] Committed changes: {final_commit_sha[:8]}")
            update_attempt_metadata(db, attempt_id, final_commit_sha=final_commit_sha)
        else:
            logging.info(f"[Attempt {attempt_id}] No changes to commit")

        # Update attempt with outcome
        update_attempt_outcome(db, attempt_id, outcome, failure_reason)

        return {
            'success': True,
            'attempt_id': attempt_id,
            'actions': actions,
            'outcome': outcome,
            'response_text': response_text
        }

    except ToolLimitExceeded as e:
        # Tool limit exceeded - mark as needs decomposition
        logging.warning(f"[Attempt {attempt_id}] Tool limit exceeded - needs decomposition")

        # Commit changes even if we hit the limit
        if 'worktree_path' in locals():
            final_commit_sha = git_manager.commit_worktree_changes(
                worktree_path,
                f"Attempt {attempt_id}: needs_decomposition (tool limit exceeded)\n\nGoal: {goal_description}"
            )
            if final_commit_sha:
                logging.info(f"[Attempt {attempt_id}] Committed changes: {final_commit_sha[:8]}")
                update_attempt_metadata(db, attempt_id, final_commit_sha=final_commit_sha)

        update_attempt_outcome(db, attempt_id, "needs_decomposition", str(e))

        return {
            'success': False,
            'attempt_id': attempt_id,
            'actions': actions,
            'outcome': "needs_decomposition",
            'error': str(e)
        }

    except Exception as e:
        # If something went wrong, record failure
        logging.error(f"[Attempt {attempt_id}] Failed with exception: {e}")

        # Commit changes even on failure
        if 'worktree_path' in locals():
            try:
                final_commit_sha = git_manager.commit_worktree_changes(
                    worktree_path,
                    f"Attempt {attempt_id}: failure\n\nGoal: {goal_description}\n\nError: {str(e)}"
                )
                if final_commit_sha:
                    logging.info(f"[Attempt {attempt_id}] Committed changes: {final_commit_sha[:8]}")
                    update_attempt_metadata(db, attempt_id, final_commit_sha=final_commit_sha)
            except Exception as commit_error:
                logging.warning(f"[Attempt {attempt_id}] Failed to commit changes: {commit_error}")

        update_attempt_outcome(db, attempt_id, "failure", str(e))

        return {
            'success': False,
            'attempt_id': attempt_id,
            'actions': actions,
            'outcome': "failure",
            'error': str(e)
        }

    finally:
        # Always clean up container
        container.stop()
        logging.debug(f"[Attempt {attempt_id}] Cleaned up container")


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

    print(f"Created session: {session_dir}")
    return session_dir, db


def load_session(session_path):
    """
    Load an existing session.
    Returns (session_dir, db, session_record).
    """
    session_dir = Path(session_path)
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        sys.exit(1)

    db_path = session_dir / "goals.db"
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    # Enable foreign key constraints
    db.execute("PRAGMA foreign_keys = ON")

    session_record = get_session_record(db)

    print(f"Loaded session: {session_dir}")
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
        '--model',
        default='openrouter/anthropic/claude-haiku-4.5',
        help='LLM model to use (default: openrouter/anthropic/claude-haiku-4.5)'
    )
    parser.add_argument(
        '--image',
        default='shots-on-goal:latest',
        help='Container image to use (default: shots-on-goal:latest)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
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
        print(f"Resuming: {session_record['description']}")
        print(f"Repo: {session_record['repo_path']}")
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
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    if not (repo_path / ".git").exists():
        print(f"Error: Not a git repository: {repo_path}")
        sys.exit(1)

    # Create new session
    session_dir, db = create_session(args.goal, str(repo_path))
    logging.info(f"Created session: {session_dir.name}")

    # Create root goal
    root_goal_id = create_goal(db, args.goal)
    logging.info(f"Created root goal (ID: {root_goal_id}): {args.goal}")

    # Start working on the root goal
    print("\n" + "=" * 80)
    print("Starting work on goal...")
    print("=" * 80 + "\n")

    try:
        result = work_on_goal(
            db=db,
            goal_id=root_goal_id,
            repo_path=str(repo_path),
            image=args.image,
            runtime=detect_container_runtime(),
            model_id=args.model
        )

        print("\n" + "=" * 80)
        print("Goal attempt completed!")
        print("=" * 80)
        print(f"\nAttempt ID: {result['attempt_id']}")
        print(f"Outcome: {result['outcome']}")
        print(f"Tools used: {len(result['actions'])}")

        if result.get('response_text'):
            print(f"\n--- LLM Response ---")
            print(result['response_text'])
            print("--- End Response ---\n")

        if result['outcome'] == 'success':
            print("✓ Goal attempt succeeded!")
        else:
            print("✗ Goal attempt failed")
            if result.get('error'):
                print(f"Error: {result['error']}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Session saved.")
        update_session_status(db, 'interrupted')
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        update_session_status(db, 'failed')
        raise
    finally:
        db.close()

    print(f"\nSession saved to: {session_dir}")


if __name__ == "__main__":
    main()

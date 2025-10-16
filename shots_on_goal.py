#!/usr/bin/env python3
"""
Shots on Goal - Autonomous goal-driven code migration system

Takes a goal and a git repository, recursively breaks down the goal,
makes changes, and validates them.
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
import json


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
            FOREIGN KEY (goal_id) REFERENCES goals(id)
        )
    """)

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


def create_attempt(db, goal_id, git_branch=None):
    """Create a new attempt for a goal and return its ID."""
    cursor = db.execute(
        "INSERT INTO attempts (goal_id, git_branch) VALUES (?, ?)",
        (goal_id, git_branch)
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


def get_actions(db, attempt_id):
    """Get all actions for an attempt."""
    cursor = db.execute(
        "SELECT * FROM actions WHERE attempt_id = ? ORDER BY executed_at",
        (attempt_id,)
    )
    return cursor.fetchall()


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

    args = parser.parse_args()

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

    # Create root goal
    root_goal_id = create_goal(db, args.goal)
    print(f"Created root goal: {args.goal} (ID: {root_goal_id})")

    # TODO: Start working on the root goal
    print("\nReady to start working on goal...")
    print("(Work loop not yet implemented)")

    db.close()


if __name__ == "__main__":
    main()

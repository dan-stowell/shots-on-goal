#!/usr/bin/env python3
"""
Tests for the V2 database schema.
Run with: python3 test_schema_v2.py
"""

import sqlite3
import json
import datetime
import sys

# Import functions from main module
from shots_on_goal import (
    init_database_v2,
    create_session_v2,
    get_session_v2,
    create_tool_v2,
    get_tool_v2,
    get_tools_for_session_v2,
    create_goal_v2,
    get_goal_v2,
    get_child_goals_v2,
    create_validation_step_v2,
    get_validation_steps_v2,
    create_branch_v2,
    get_branch_v2,
    get_branch_by_name_v2,
    create_worktree_v2,
    get_worktree_v2,
    create_attempt_v2,
    get_attempt_v2,
    get_attempts_for_goal_v2,
    create_attempt_tool_v2,
    get_tools_for_attempt_v2,
    create_attempt_result_v2,
    get_attempt_result_v2,
    get_attempt_result_by_attempt_v2,
    create_tool_call_v2,
    get_tool_calls_v2,
    create_validation_run_v2,
    get_validation_runs_v2,
    create_merge_v2,
    get_merge_v2,
)


def run_test(test_name, test_func):
    """Run a single test and report results."""
    try:
        db = init_database_v2(':memory:')
        test_func(db)
        print(f"✓ {test_name}")
        return True
    except AssertionError as e:
        print(f"✗ {test_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ {test_name}: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_tables(db):
    """Verify all V2 tables are created."""
    cursor = db.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """)
    tables = [row[0] for row in cursor.fetchall()]

    expected_tables = [
        'attempt',
        'attempt_result',
        'attempt_tool',
        'branch',
        'goal',
        'merge',
        'session',
        'tool',
        'tool_call',
        'validation_run',
        'validation_step',
        'worktree',
    ]

    for table in expected_tables:
        assert table in tables, f"Table '{table}' not found"


def test_foreign_keys_enabled(db):
    """Verify foreign key constraints are enabled."""
    cursor = db.cursor()
    cursor.execute("PRAGMA foreign_keys")
    result = cursor.fetchone()
    assert result[0] == 1, "Foreign keys should be enabled"


def test_session_crud(db):
    """Test session create and read operations."""
    # Create session
    session_id = create_session_v2(
        db,
        initial_goal="Test goal",
        model_a="model-a",
        model_b="model-b",
        flags={"max_tools": 50},
        repo_path="/tmp/repo",
        base_branch="main"
    )

    assert session_id is not None
    assert session_id > 0

    # Read session back
    session = get_session_v2(db, session_id)
    assert session is not None
    assert session['initial_goal'] == "Test goal"
    assert session['model_a'] == "model-a"
    assert session['model_b'] == "model-b"
    assert session['repo_path'] == "/tmp/repo"
    assert session['base_branch'] == "main"

    # Verify flags are stored as JSON
    flags = json.loads(session['flags'])
    assert flags['max_tools'] == 50


def test_tool_crud(db):
    """Test tool create and read operations."""
    # Create session first
    session_id = create_session_v2(
        db, "Test", "a", "b", {}, "/tmp", "main"
    )

    # Create tool
    tool_id = create_tool_v2(db, session_id, "read_file", "Read a file")
    assert tool_id is not None

    # Get tool
    tool = get_tool_v2(db, tool_id)
    assert tool is not None
    assert tool['name'] == "read_file"
    assert tool['description'] == "Read a file"
    assert tool['session_id'] == session_id

    # Create more tools
    create_tool_v2(db, session_id, "write_file", "Write a file")
    create_tool_v2(db, session_id, "list_directory", "List directory")

    # Get all tools for session
    tools = get_tools_for_session_v2(db, session_id)
    assert len(tools) == 3
    assert {t['name'] for t in tools} == {"read_file", "write_file", "list_directory"}


def test_goal_crud(db):
    """Test goal create and read operations."""
    # Create session
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")

    # Create root goal
    goal_id = create_goal_v2(
        db,
        session_id=session_id,
        goal_text="Migrate to Bazel",
        source='cli'
    )
    assert goal_id is not None

    # Get goal
    goal = get_goal_v2(db, goal_id)
    assert goal is not None
    assert goal['goal_text'] == "Migrate to Bazel"
    assert goal['parent_goal_id'] is None
    assert goal['source'] == 'cli'
    assert goal['created_by_attempt_id'] is None


def test_goal_hierarchy(db):
    """Test parent-child goal relationships."""
    # Create session
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")

    # Create parent goal
    parent_id = create_goal_v2(
        db,
        session_id=session_id,
        goal_text="Parent goal",
        source='cli'
    )

    # Create child goals
    child1_id = create_goal_v2(
        db,
        session_id=session_id,
        goal_text="Child goal 1",
        parent_goal_id=parent_id,
        order_num=1,
        source='breakdown'
    )

    child2_id = create_goal_v2(
        db,
        session_id=session_id,
        goal_text="Child goal 2",
        parent_goal_id=parent_id,
        order_num=2,
        source='breakdown'
    )

    # Get child goals
    children = get_child_goals_v2(db, parent_id)
    assert len(children) == 2
    assert children[0]['goal_text'] == "Child goal 1"
    assert children[1]['goal_text'] == "Child goal 2"
    assert children[0]['order_num'] == 1
    assert children[1]['order_num'] == 2


def test_validation_steps(db):
    """Test validation step create and read operations."""
    # Create session and goal
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    goal_id = create_goal_v2(db, session_id, "Test goal", source='cli')

    # Create validation steps
    step1_id = create_validation_step_v2(
        db, goal_id, order_num=1, command="bazel build //...", source='cli'
    )
    step2_id = create_validation_step_v2(
        db, goal_id, order_num=2, command="bazel test //...", source='cli'
    )

    # Get validation steps
    steps = get_validation_steps_v2(db, goal_id)
    assert len(steps) == 2
    assert steps[0]['command'] == "bazel build //..."
    assert steps[1]['command'] == "bazel test //..."
    assert steps[0]['order_num'] == 1
    assert steps[1]['order_num'] == 2


def test_foreign_key_constraints(db):
    """Test that foreign key constraints are enforced."""
    # Try to create a goal with non-existent session_id
    # This should fail with foreign key constraint
    try:
        create_goal_v2(db, session_id=999, goal_text="Invalid", source='cli')
        assert False, "Should have raised IntegrityError"
    except sqlite3.IntegrityError:
        pass  # Expected


def test_timestamps_auto_created(db):
    """Verify timestamps are automatically created."""
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    session = get_session_v2(db, session_id)

    assert session['timestamp'] is not None
    # Timestamp should be a recent time (rough check)
    ts = datetime.datetime.fromisoformat(session['timestamp'])
    now = datetime.datetime.now()
    diff = (now - ts).total_seconds()
    assert diff < 5, "Timestamp should be recent"


def test_branch_crud(db):
    """Test branch create and read operations."""
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    goal_id = create_goal_v2(db, session_id, "Test goal", source='cli')

    # Create base branch
    base_branch_id = create_branch_v2(
        db,
        session_id=session_id,
        name="main",
        reason="base branch",
        created_by_goal_id=goal_id
    )

    # Create child branch
    child_branch_id = create_branch_v2(
        db,
        session_id=session_id,
        name="feature-branch",
        parent_branch_id=base_branch_id,
        parent_commit_sha="abc123",
        reason="feature work",
        created_by_goal_id=goal_id
    )

    # Verify base branch
    base_branch = get_branch_v2(db, base_branch_id)
    assert base_branch is not None
    assert base_branch['name'] == "main"
    assert base_branch['parent_branch_id'] is None

    # Verify child branch
    child_branch = get_branch_v2(db, child_branch_id)
    assert child_branch is not None
    assert child_branch['name'] == "feature-branch"
    assert child_branch['parent_branch_id'] == base_branch_id
    assert child_branch['parent_commit_sha'] == "abc123"

    # Get by name
    found = get_branch_by_name_v2(db, session_id, "feature-branch")
    assert found is not None
    assert found['id'] == child_branch_id


def test_worktree_crud(db):
    """Test worktree create and read operations."""
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    goal_id = create_goal_v2(db, session_id, "Test goal", source='cli')
    branch_id = create_branch_v2(db, session_id, "test-branch", created_by_goal_id=goal_id)

    # Create worktree
    worktree_id = create_worktree_v2(
        db,
        branch_id=branch_id,
        path="/tmp/worktrees/test-1",
        start_sha="def456",
        reason="attempt 1"
    )

    # Verify worktree
    worktree = get_worktree_v2(db, worktree_id)
    assert worktree is not None
    assert worktree['branch_id'] == branch_id
    assert worktree['path'] == "/tmp/worktrees/test-1"
    assert worktree['start_sha'] == "def456"


def test_attempt_lifecycle(db):
    """Test complete attempt lifecycle: start -> tool calls -> result."""
    # Setup
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    goal_id = create_goal_v2(db, session_id, "Test goal", source='cli')
    branch_id = create_branch_v2(db, session_id, "test-branch", created_by_goal_id=goal_id)
    worktree_id = create_worktree_v2(db, branch_id, "/tmp/wt", "abc123")

    # Create tools
    tool1_id = create_tool_v2(db, session_id, "read_file", "Read file")
    tool2_id = create_tool_v2(db, session_id, "write_file", "Write file")

    # Create attempt
    attempt_id = create_attempt_v2(
        db,
        goal_id=goal_id,
        worktree_id=worktree_id,
        start_commit_sha="abc123",
        prompt="Do the thing",
        model="test-model",
        attempt_type="implementation"
    )

    assert attempt_id is not None

    # Link tools to attempt
    create_attempt_tool_v2(db, attempt_id, tool1_id)
    create_attempt_tool_v2(db, attempt_id, tool2_id)

    # Verify tools
    tools = get_tools_for_attempt_v2(db, attempt_id)
    assert len(tools) == 2
    assert {t['name'] for t in tools} == {"read_file", "write_file"}

    # Record tool calls
    create_tool_call_v2(db, attempt_id, 1, tool1_id, "read_file", '{"path": "foo.txt"}', "contents")
    create_tool_call_v2(db, attempt_id, 2, tool2_id, "write_file", '{"path": "bar.txt"}', "ok")
    create_tool_call_v2(db, attempt_id, 3, tool1_id, "read_file", '{"path": "bar.txt"}', "contents2")

    # Verify tool calls
    calls = get_tool_calls_v2(db, attempt_id)
    assert len(calls) == 3
    assert calls[0]['order_num'] == 1
    assert calls[0]['tool_name'] == "read_file"
    assert calls[1]['order_num'] == 2
    assert calls[2]['order_num'] == 3

    # Create attempt result
    result_id = create_attempt_result_v2(
        db,
        attempt_id=attempt_id,
        end_commit_sha="def456",
        diff="some diff",
        status="success",
        status_detail=None
    )

    # Verify attempt result
    result = get_attempt_result_v2(db, result_id)
    assert result is not None
    assert result['status'] == "success"
    assert result['end_commit_sha'] == "def456"

    # Get result by attempt
    result2 = get_attempt_result_by_attempt_v2(db, attempt_id)
    assert result2 is not None
    assert result2['id'] == result_id


def test_validation_runs(db):
    """Test validation run lifecycle."""
    # Setup
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    goal_id = create_goal_v2(db, session_id, "Test goal", source='cli')
    branch_id = create_branch_v2(db, session_id, "test-branch", created_by_goal_id=goal_id)
    worktree_id = create_worktree_v2(db, branch_id, "/tmp/wt", "abc123")

    # Create validation steps
    step1_id = create_validation_step_v2(db, goal_id, 1, "bazel build //...", "cli")
    step2_id = create_validation_step_v2(db, goal_id, 2, "bazel test //...", "cli")

    # Create attempt and result
    attempt_id = create_attempt_v2(db, goal_id, worktree_id, "abc", "prompt", "model", "implementation")
    result_id = create_attempt_result_v2(db, attempt_id, "def", "diff", "completed")

    # Run validations
    run1_id = create_validation_run_v2(db, step1_id, result_id, 0, "build output")
    run2_id = create_validation_run_v2(db, step2_id, result_id, 1, "test failed")

    # Verify validation runs
    runs = get_validation_runs_v2(db, result_id)
    assert len(runs) == 2
    assert runs[0]['exit_code'] == 0  # build passed
    assert runs[1]['exit_code'] == 1  # test failed


def test_merge_crud(db):
    """Test merge create and read operations."""
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    goal_id = create_goal_v2(db, session_id, "Test goal", source='cli')

    # Create branches
    from_branch_id = create_branch_v2(db, session_id, "feature", created_by_goal_id=goal_id)
    to_branch_id = create_branch_v2(db, session_id, "main", created_by_goal_id=goal_id)

    # Create merge
    merge_id = create_merge_v2(
        db,
        from_branch_id=from_branch_id,
        from_commit_sha="abc123",
        to_branch_id=to_branch_id,
        to_commit_sha="def456",
        result_commit_sha="ghi789"
    )

    # Verify merge
    merge = get_merge_v2(db, merge_id)
    assert merge is not None
    assert merge['from_branch_id'] == from_branch_id
    assert merge['to_branch_id'] == to_branch_id
    assert merge['result_commit_sha'] == "ghi789"


def test_multiple_attempts_per_goal(db):
    """Test that a goal can have multiple attempts."""
    session_id = create_session_v2(db, "Test", "a", "b", {}, "/tmp", "main")
    goal_id = create_goal_v2(db, session_id, "Test goal", source='cli')
    branch_id = create_branch_v2(db, session_id, "test-branch", created_by_goal_id=goal_id)

    # Create 3 attempts
    wt1 = create_worktree_v2(db, branch_id, "/tmp/wt1", "sha1")
    wt2 = create_worktree_v2(db, branch_id, "/tmp/wt2", "sha2")
    wt3 = create_worktree_v2(db, branch_id, "/tmp/wt3", "sha3")

    attempt1 = create_attempt_v2(db, goal_id, wt1, "sha1", "prompt", "model-a", "implementation")
    attempt2 = create_attempt_v2(db, goal_id, wt2, "sha2", "prompt", "model-b", "implementation")
    attempt3 = create_attempt_v2(db, goal_id, wt3, "sha3", "prompt", "model-a", "breakdown")

    # Get all attempts for goal
    attempts = get_attempts_for_goal_v2(db, goal_id)
    assert len(attempts) == 3
    assert attempts[0]['id'] == attempt1
    assert attempts[1]['id'] == attempt2
    assert attempts[2]['id'] == attempt3
    assert attempts[2]['attempt_type'] == "breakdown"


def main():
    """Run all tests."""
    print("=" * 70)
    print("Running V2 Schema Tests (Phases 1-4)")
    print("=" * 70)

    tests = [
        # Phase 1: Schema & core infrastructure
        ("Create tables", test_create_tables),
        ("Foreign keys enabled", test_foreign_keys_enabled),
        ("Session CRUD", test_session_crud),
        ("Tool CRUD", test_tool_crud),
        ("Goal CRUD", test_goal_crud),
        ("Goal hierarchy", test_goal_hierarchy),
        ("Validation steps", test_validation_steps),
        ("Foreign key constraints", test_foreign_key_constraints),
        ("Timestamps auto-created", test_timestamps_auto_created),
        # Phase 2 & 3: Attempt lifecycle & git operations
        ("Branch CRUD", test_branch_crud),
        ("Worktree CRUD", test_worktree_crud),
        ("Attempt lifecycle", test_attempt_lifecycle),
        ("Multiple attempts per goal", test_multiple_attempts_per_goal),
        # Phase 4: Validation runs
        ("Validation runs", test_validation_runs),
        ("Merge CRUD", test_merge_crud),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

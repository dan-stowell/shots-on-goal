#!/usr/bin/env python3
"""Unit tests for the V2 database schema."""

import json
import sqlite3
import unittest
from datetime import datetime

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


class SchemaV2TestCase(unittest.TestCase):
    """Test the V2 schema helpers using an in-memory database."""

    def setUp(self):
        self.db = init_database_v2(':memory:')

    def tearDown(self):
        self.db.close()

    def test_tables_created(self):
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
        """)
        tables = {row[0] for row in cursor.fetchall()}
        expected = {
            'session', 'tool', 'goal', 'validation_step', 'branch',
            'worktree', 'attempt', 'attempt_tool', 'attempt_result',
            'tool_call', 'validation_run', 'merge'
        }
        self.assertTrue(expected.issubset(tables))

    def test_foreign_keys_enabled(self):
        cursor = self.db.cursor()
        cursor.execute("PRAGMA foreign_keys")
        self.assertEqual(cursor.fetchone()[0], 1)

    def test_session_crud(self):
        session_id = create_session_v2(
            self.db,
            initial_goal="Test goal",
            model_a="model-a",
            model_b="model-b",
            flags={"max_tools": 50},
            repo_path="/tmp/repo",
            base_branch="main"
        )
        session = get_session_v2(self.db, session_id)
        self.assertEqual(session['initial_goal'], "Test goal")
        self.assertEqual(session['model_a'], "model-a")
        flags = json.loads(session['flags'])
        self.assertEqual(flags['max_tools'], 50)
        ts = datetime.fromisoformat(session['timestamp'])
        self.assertLess((datetime.now() - ts).total_seconds(), 5)

    def test_tool_crud(self):
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        tool_id = create_tool_v2(self.db, session_id, "read_file", "Read a file")
        tool = get_tool_v2(self.db, tool_id)
        self.assertEqual(tool['name'], "read_file")
        tools = get_tools_for_session_v2(self.db, session_id)
        self.assertEqual(len(tools), 1)

    def test_goal_hierarchy(self):
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        parent = create_goal_v2(self.db, session_id, "Parent", source='cli')
        create_goal_v2(self.db, session_id, "Child 1", parent_goal_id=parent, order_num=1, source='breakdown')
        create_goal_v2(self.db, session_id, "Child 2", parent_goal_id=parent, order_num=2, source='breakdown')
        children = get_child_goals_v2(self.db, parent)
        self.assertEqual([c['goal_text'] for c in children], ["Child 1", "Child 2"])

    def test_validation_steps(self):
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        create_validation_step_v2(self.db, goal_id, 1, "bazel build //...")
        steps = get_validation_steps_v2(self.db, goal_id)
        self.assertEqual(len(steps), 1)

    def test_branch_and_worktree(self):
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        branch_id = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)
        worktree_id = create_worktree_v2(self.db, branch_id, "/tmp/wt", "abc123")
        branch = get_branch_v2(self.db, branch_id)
        self.assertEqual(branch['name'], "feature")
        worktree = get_worktree_v2(self.db, worktree_id)
        self.assertEqual(worktree['start_sha'], "abc123")

    def test_attempt_lifecycle(self):
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        branch_id = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)
        worktree_id = create_worktree_v2(self.db, branch_id, "/tmp/wt", "abc123")
        attempt_id = create_attempt_v2(self.db, goal_id, worktree_id, "abc123", "prompt", "model", "implementation")
        result_id = create_attempt_result_v2(self.db, attempt_id, "def456", "diff", "success")
        attempt = get_attempt_v2(self.db, attempt_id)
        self.assertEqual(attempt['attempt_type'], "implementation")
        result = get_attempt_result_v2(self.db, result_id)
        self.assertEqual(result['status'], "success")

    def test_tool_calls_and_validation_runs(self):
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        tool_id = create_tool_v2(self.db, session_id, "read_file", "")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        branch_id = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)
        worktree_id = create_worktree_v2(self.db, branch_id, "/tmp/wt", "abc123")
        attempt_id = create_attempt_v2(self.db, goal_id, worktree_id, "abc123", "prompt", "model", "implementation")
        create_attempt_tool_v2(self.db, attempt_id, tool_id)
        create_tool_call_v2(self.db, attempt_id, 1, tool_id, "read_file", "{}", "ok")
        calls = get_tool_calls_v2(self.db, attempt_id)
        self.assertEqual(len(calls), 1)
        step_id = create_validation_step_v2(self.db, goal_id, 1, "bazel build //...")
        result_id = create_attempt_result_v2(self.db, attempt_id, "def", "diff", "completed")
        create_validation_run_v2(self.db, step_id, result_id, 0, "output")
        runs = get_validation_runs_v2(self.db, result_id)
        self.assertEqual(len(runs), 1)

    def test_merge_crud(self):
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        from_branch = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)
        to_branch = create_branch_v2(self.db, session_id, "main", created_by_goal_id=goal_id)
        merge_id = create_merge_v2(self.db, from_branch, "abc", to_branch, "def", "ghi")
        merge = get_merge_v2(self.db, merge_id)
        self.assertEqual(merge['result_commit_sha'], "ghi")


if __name__ == '__main__':  # pragma: no cover
    unittest.main(verbosity=2)

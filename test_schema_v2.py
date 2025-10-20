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
    # Query helpers
    get_goal_with_attempts_v2,
    get_attempt_with_details_v2,
    get_goal_tree_v2,
    get_validation_status_v2,
    get_goal_ancestry_v2,
    get_attempts_summary_v2,
    # Orchestration
    work_on_goal_v2_simple,
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

    def test_get_goal_with_attempts(self):
        """Test getting goal with all attempts."""
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        branch_id = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)
        wt1 = create_worktree_v2(self.db, branch_id, "/tmp/wt1", "abc")
        wt2 = create_worktree_v2(self.db, branch_id, "/tmp/wt2", "def")

        # Create 2 attempts
        attempt1 = create_attempt_v2(self.db, goal_id, wt1, "abc", "prompt", "model-a", "implementation")
        create_attempt_result_v2(self.db, attempt1, "abc2", "diff", "success")

        attempt2 = create_attempt_v2(self.db, goal_id, wt2, "def", "prompt", "model-b", "implementation")
        create_attempt_result_v2(self.db, attempt2, "def2", "diff", "error", "timeout")

        # Get goal with attempts
        data = get_goal_with_attempts_v2(self.db, goal_id)
        self.assertIsNotNone(data)
        self.assertEqual(data['goal']['id'], goal_id)
        self.assertEqual(len(data['attempts']), 2)
        self.assertEqual(data['attempts'][0]['result']['status'], "success")
        self.assertEqual(data['attempts'][1]['result']['status'], "error")

    def test_get_attempt_with_details(self):
        """Test getting attempt with all related data."""
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        tool_id = create_tool_v2(self.db, session_id, "read_file", "")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        branch_id = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)
        worktree_id = create_worktree_v2(self.db, branch_id, "/tmp/wt", "abc")
        attempt_id = create_attempt_v2(self.db, goal_id, worktree_id, "abc", "prompt", "model", "implementation")

        # Add tool and tool call
        create_attempt_tool_v2(self.db, attempt_id, tool_id)
        create_tool_call_v2(self.db, attempt_id, 1, tool_id, "read_file", "{}", "ok")

        # Add result
        result_id = create_attempt_result_v2(self.db, attempt_id, "def", "diff", "success")

        # Get details
        details = get_attempt_with_details_v2(self.db, attempt_id)
        self.assertIsNotNone(details)
        self.assertEqual(details['attempt']['id'], attempt_id)
        self.assertEqual(details['goal']['id'], goal_id)
        self.assertEqual(details['branch']['id'], branch_id)
        self.assertEqual(len(details['tools']), 1)
        self.assertEqual(len(details['tool_calls']), 1)
        self.assertEqual(details['result']['status'], "success")

    def test_get_goal_tree(self):
        """Test getting goal hierarchy recursively."""
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")

        # Create hierarchy: root -> child1, child2 -> grandchild1
        root = create_goal_v2(self.db, session_id, "Root")
        child1 = create_goal_v2(self.db, session_id, "Child 1", parent_goal_id=root, order_num=1)
        child2 = create_goal_v2(self.db, session_id, "Child 2", parent_goal_id=root, order_num=2)
        grandchild1 = create_goal_v2(self.db, session_id, "Grandchild 1", parent_goal_id=child2, order_num=1)

        # Get tree
        tree = get_goal_tree_v2(self.db, root)
        self.assertEqual(tree['goal']['goal_text'], "Root")
        self.assertEqual(len(tree['children']), 2)
        self.assertEqual(tree['children'][0]['goal']['goal_text'], "Child 1")
        self.assertEqual(tree['children'][1]['goal']['goal_text'], "Child 2")
        self.assertEqual(len(tree['children'][1]['children']), 1)
        self.assertEqual(tree['children'][1]['children'][0]['goal']['goal_text'], "Grandchild 1")

    def test_get_validation_status(self):
        """Test checking validation status."""
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        goal_id = create_goal_v2(self.db, session_id, "Goal")

        # Create validation steps
        step1 = create_validation_step_v2(self.db, goal_id, 1, "bazel build //...")
        step2 = create_validation_step_v2(self.db, goal_id, 2, "bazel test //...")

        # Create attempt and result
        branch_id = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)
        worktree_id = create_worktree_v2(self.db, branch_id, "/tmp/wt", "abc")
        attempt_id = create_attempt_v2(self.db, goal_id, worktree_id, "abc", "prompt", "model", "implementation")
        result_id = create_attempt_result_v2(self.db, attempt_id, "def", "diff", "success")

        # Run validations (first passes, second fails)
        create_validation_run_v2(self.db, step1, result_id, 0, "build ok")
        create_validation_run_v2(self.db, step2, result_id, 1, "test failed")

        # Check status
        status = get_validation_status_v2(self.db, goal_id)
        self.assertTrue(status['has_validation'])
        self.assertFalse(status['all_passed'])
        self.assertEqual(status['passed_count'], 1)
        self.assertEqual(status['failed_count'], 1)
        self.assertTrue(status['steps'][0]['passed'])
        self.assertFalse(status['steps'][1]['passed'])

    def test_get_goal_ancestry(self):
        """Test getting goal ancestry chain."""
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")

        # Create chain: root -> parent -> child
        root = create_goal_v2(self.db, session_id, "Root")
        parent = create_goal_v2(self.db, session_id, "Parent", parent_goal_id=root)
        child = create_goal_v2(self.db, session_id, "Child", parent_goal_id=parent)

        # Get ancestry
        ancestry = get_goal_ancestry_v2(self.db, child)
        self.assertEqual(len(ancestry), 3)
        self.assertEqual(ancestry[0]['goal_text'], "Root")
        self.assertEqual(ancestry[1]['goal_text'], "Parent")
        self.assertEqual(ancestry[2]['goal_text'], "Child")

    def test_get_attempts_summary(self):
        """Test getting summary statistics for attempts."""
        session_id = create_session_v2(self.db, "Test", "a", "b", {}, "/tmp", "main")
        tool_id = create_tool_v2(self.db, session_id, "read_file", "")
        goal_id = create_goal_v2(self.db, session_id, "Goal")
        branch_id = create_branch_v2(self.db, session_id, "feature", created_by_goal_id=goal_id)

        # Create 3 attempts with different models and outcomes
        for i, (model, status) in enumerate([("model-a", "success"), ("model-b", "error"), ("model-a", "timeout")]):
            wt = create_worktree_v2(self.db, branch_id, f"/tmp/wt{i}", "abc")
            attempt = create_attempt_v2(self.db, goal_id, wt, "abc", "prompt", model, "implementation")
            create_tool_call_v2(self.db, attempt, 1, tool_id, "read_file", "{}", "ok")
            create_tool_call_v2(self.db, attempt, 2, tool_id, "read_file", "{}", "ok")
            create_attempt_result_v2(self.db, attempt, "def", "diff", status)

        # Get summary
        summary = get_attempts_summary_v2(self.db, goal_id)
        self.assertEqual(summary['total_attempts'], 3)
        self.assertEqual(summary['by_model']['model-a'], 2)
        self.assertEqual(summary['by_model']['model-b'], 1)
        self.assertEqual(summary['by_type']['implementation'], 3)
        self.assertEqual(summary['by_status']['success'], 1)
        self.assertEqual(summary['by_status']['error'], 1)
        self.assertEqual(summary['by_status']['timeout'], 1)
        self.assertEqual(summary['total_tool_calls'], 6)  # 2 calls per attempt * 3 attempts

    def test_work_on_goal_v2_simple(self):
        """Test simplified V2 orchestration end-to-end."""
        import tempfile
        import subprocess
        import os

        # Create temp git repo
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(['git', 'init'], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=tmpdir, check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=tmpdir, check=True)

            # Create initial commit
            readme_path = os.path.join(tmpdir, 'README.md')
            with open(readme_path, 'w') as f:
                f.write('# Test Repo\n')
            subprocess.run(['git', 'add', '.'], cwd=tmpdir, check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=tmpdir, check=True, capture_output=True)

            # Create session and goal
            session_id = create_session_v2(self.db, "Test goal", "model-a", "model-b", {}, tmpdir, "main")

            # Create some tools
            tool1 = create_tool_v2(self.db, session_id, "read_file", "Read a file")
            tool2 = create_tool_v2(self.db, session_id, "write_file", "Write a file")

            # Create goal with validation
            goal_id = create_goal_v2(self.db, session_id, "Add BUILD file")
            create_validation_step_v2(self.db, goal_id, 1, "test -f BUILD", "cli")

            # Run orchestration
            result = work_on_goal_v2_simple(
                self.db,
                session_id=session_id,
                goal_id=goal_id,
                repo_path=tmpdir,
                model_id="test-model",
                attempt_type="implementation"
            )

            # Verify result
            self.assertTrue(result['success'])
            self.assertIsNotNone(result['attempt_id'])
            self.assertIsNotNone(result['result_id'])
            self.assertTrue(result['validation_passed'])

            # Verify data was recorded
            attempt = get_attempt_v2(self.db, result['attempt_id'])
            self.assertEqual(attempt['model'], "test-model")
            self.assertEqual(attempt['attempt_type'], "implementation")

            # Verify tool calls were recorded
            tool_calls = get_tool_calls_v2(self.db, result['attempt_id'])
            self.assertGreater(len(tool_calls), 0)

            # Verify validation runs
            validation_status = get_validation_status_v2(self.db, goal_id)
            self.assertTrue(validation_status['has_validation'])
            self.assertTrue(validation_status['all_passed'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main(verbosity=2)

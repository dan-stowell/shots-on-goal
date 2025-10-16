#!/usr/bin/env python3
"""
Test suite for shots_on_goal.py
Run with: python3 test_shots_on_goal.py
"""

import unittest
import tempfile
import shutil
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

# Import functions from main module
import shots_on_goal


class TestDatabase(unittest.TestCase):
    """Test database schema and operations"""

    def setUp(self):
        """Create temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = shots_on_goal.init_database(str(self.db_path))

    def tearDown(self):
        """Clean up temporary files"""
        self.db.close()
        shutil.rmtree(self.temp_dir)

    def test_database_initialization(self):
        """Test that all tables are created"""
        cursor = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        self.assertIn('session', tables)
        self.assertIn('goals', tables)
        self.assertIn('attempts', tables)
        self.assertIn('actions', tables)

    def test_foreign_keys_enabled(self):
        """Test that foreign key constraints are enabled"""
        cursor = self.db.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()[0]
        self.assertEqual(result, 1, "Foreign keys should be enabled")

    def test_foreign_key_constraint_attempts_to_goals(self):
        """Test that attempts.goal_id references goals.id"""
        cursor = self.db.execute("SELECT * FROM pragma_foreign_key_list('attempts')")
        fk_info = cursor.fetchall()

        self.assertEqual(len(fk_info), 1, "Should have exactly one foreign key")
        # fk_info format: (id, seq, table, from, to, on_update, on_delete, match)
        self.assertEqual(fk_info[0][2], 'goals', "FK should reference goals table")
        self.assertEqual(fk_info[0][3], 'goal_id', "FK should be on goal_id column")
        self.assertEqual(fk_info[0][4], 'id', "FK should reference id column")

    def test_foreign_key_constraint_enforcement(self):
        """Test that invalid goal_id is rejected"""
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("INSERT INTO attempts (goal_id) VALUES (999)")
            self.db.commit()

    def test_session_single_row_constraint(self):
        """Test that only one session row can exist"""
        # Insert first row (should succeed)
        shots_on_goal.create_session_record(
            self.db, 'test-123', 'Test session', '/tmp/repo'
        )

        # Try to insert second row (should fail)
        with self.assertRaises(sqlite3.IntegrityError):
            shots_on_goal.create_session_record(
                self.db, 'test-456', 'Another session', '/tmp/repo2'
            )


class TestSessionRecord(unittest.TestCase):
    """Test session record operations"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = shots_on_goal.init_database(str(self.db_path))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir)

    def test_create_session_record(self):
        """Test creating a session record"""
        session_id = '20251016-103000'
        description = 'Test migration'
        repo_path = '/tmp/test-repo'

        shots_on_goal.create_session_record(
            self.db, session_id, description, repo_path
        )

        session = shots_on_goal.get_session_record(self.db)
        self.assertIsNotNone(session)
        self.assertEqual(session['session_id'], session_id)
        self.assertEqual(session['description'], description)
        self.assertEqual(session['repo_path'], repo_path)
        self.assertEqual(session['base_branch'], 'main')
        self.assertEqual(session['status'], 'active')

    def test_update_session_status(self):
        """Test updating session status"""
        shots_on_goal.create_session_record(
            self.db, 'test-123', 'Test', '/tmp/repo'
        )

        shots_on_goal.update_session_status(self.db, 'completed')

        session = shots_on_goal.get_session_record(self.db)
        self.assertEqual(session['status'], 'completed')
        self.assertIsNotNone(session['completed_at'])


class TestGoals(unittest.TestCase):
    """Test goal operations"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = shots_on_goal.init_database(str(self.db_path))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir)

    def test_create_root_goal(self):
        """Test creating a root goal"""
        goal_id = shots_on_goal.create_goal(self.db, "Migrate to Bazel")

        goal = shots_on_goal.get_goal(self.db, goal_id)
        self.assertIsNotNone(goal)
        self.assertEqual(goal['description'], "Migrate to Bazel")
        self.assertEqual(goal['goal_type'], 'implementation')
        self.assertEqual(goal['status'], 'pending')
        self.assertIsNone(goal['parent_id'])

    def test_create_child_goal(self):
        """Test creating child goals"""
        parent_id = shots_on_goal.create_goal(self.db, "Parent goal")
        child_id = shots_on_goal.create_goal(
            self.db, "Child goal", parent_id=parent_id
        )

        child = shots_on_goal.get_goal(self.db, child_id)
        self.assertEqual(child['parent_id'], parent_id)

    def test_get_child_goals(self):
        """Test retrieving child goals"""
        parent_id = shots_on_goal.create_goal(self.db, "Parent")
        child1_id = shots_on_goal.create_goal(self.db, "Child 1", parent_id=parent_id)
        child2_id = shots_on_goal.create_goal(self.db, "Child 2", parent_id=parent_id)

        children = shots_on_goal.get_child_goals(self.db, parent_id)
        self.assertEqual(len(children), 2)
        child_ids = [c['id'] for c in children]
        self.assertIn(child1_id, child_ids)
        self.assertIn(child2_id, child_ids)

    def test_get_root_goal(self):
        """Test retrieving root goal"""
        root_id = shots_on_goal.create_goal(self.db, "Root goal")
        shots_on_goal.create_goal(self.db, "Child goal", parent_id=root_id)

        root = shots_on_goal.get_root_goal(self.db)
        self.assertEqual(root['id'], root_id)

    def test_update_goal_status(self):
        """Test updating goal status"""
        goal_id = shots_on_goal.create_goal(self.db, "Test goal")

        shots_on_goal.update_goal_status(self.db, goal_id, 'in_progress')
        goal = shots_on_goal.get_goal(self.db, goal_id)
        self.assertEqual(goal['status'], 'in_progress')

        shots_on_goal.update_goal_status(self.db, goal_id, 'completed')
        goal = shots_on_goal.get_goal(self.db, goal_id)
        self.assertEqual(goal['status'], 'completed')
        self.assertIsNotNone(goal['completed_at'])

    def test_create_decomposition_goal(self):
        """Test creating a decomposition goal"""
        goal_id = shots_on_goal.create_goal(
            self.db, "Decompose: Migrate to Bazel", goal_type='decomposition'
        )

        goal = shots_on_goal.get_goal(self.db, goal_id)
        self.assertEqual(goal['goal_type'], 'decomposition')


class TestAttempts(unittest.TestCase):
    """Test attempt operations"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = shots_on_goal.init_database(str(self.db_path))
        self.goal_id = shots_on_goal.create_goal(self.db, "Test goal")

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir)

    def test_create_attempt(self):
        """Test creating an attempt"""
        attempt_id = shots_on_goal.create_attempt(self.db, self.goal_id)

        attempts = shots_on_goal.get_attempts(self.db, self.goal_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['id'], attempt_id)
        self.assertEqual(attempts[0]['goal_id'], self.goal_id)
        self.assertIsNone(attempts[0]['outcome'])

    def test_update_attempt_outcome(self):
        """Test updating attempt outcome"""
        attempt_id = shots_on_goal.create_attempt(self.db, self.goal_id)

        shots_on_goal.update_attempt_outcome(self.db, attempt_id, 'success')

        attempts = shots_on_goal.get_attempts(self.db, self.goal_id)
        self.assertEqual(attempts[0]['outcome'], 'success')
        self.assertIsNone(attempts[0]['failure_reason'])
        self.assertIsNotNone(attempts[0]['completed_at'])

    def test_update_attempt_failure(self):
        """Test updating attempt with failure"""
        attempt_id = shots_on_goal.create_attempt(self.db, self.goal_id)

        shots_on_goal.update_attempt_outcome(
            self.db, attempt_id, 'failure', 'Tests failed'
        )

        attempts = shots_on_goal.get_attempts(self.db, self.goal_id)
        self.assertEqual(attempts[0]['outcome'], 'failure')
        self.assertEqual(attempts[0]['failure_reason'], 'Tests failed')

    def test_multiple_attempts(self):
        """Test multiple attempts for same goal"""
        attempt1_id = shots_on_goal.create_attempt(self.db, self.goal_id)
        attempt2_id = shots_on_goal.create_attempt(self.db, self.goal_id)

        attempts = shots_on_goal.get_attempts(self.db, self.goal_id)
        self.assertEqual(len(attempts), 2)

        attempt_ids = [a['id'] for a in attempts]
        self.assertIn(attempt1_id, attempt_ids)
        self.assertIn(attempt2_id, attempt_ids)

    def test_create_attempt_with_metadata(self):
        """Test creating attempt with git sha, worktree, and container info"""
        attempt_id = shots_on_goal.create_attempt(
            self.db,
            self.goal_id,
            git_branch='goal-1-attempt-1',
            worktree_path='/tmp/worktrees/goal-1-attempt-1',
            container_id='abc123def456',
            git_commit_sha='1234567890abcdef1234567890abcdef12345678'
        )

        attempts = shots_on_goal.get_attempts(self.db, self.goal_id)
        self.assertEqual(len(attempts), 1)

        attempt = attempts[0]
        self.assertEqual(attempt['git_branch'], 'goal-1-attempt-1')
        self.assertEqual(attempt['worktree_path'], '/tmp/worktrees/goal-1-attempt-1')
        self.assertEqual(attempt['container_id'], 'abc123def456')
        self.assertEqual(attempt['git_commit_sha'], '1234567890abcdef1234567890abcdef12345678')


class TestActions(unittest.TestCase):
    """Test action recording"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = shots_on_goal.init_database(str(self.db_path))
        self.goal_id = shots_on_goal.create_goal(self.db, "Test goal")
        self.attempt_id = shots_on_goal.create_attempt(self.db, self.goal_id)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir)

    def test_record_action(self):
        """Test recording a tool action"""
        action_id = shots_on_goal.record_action(
            self.db, self.attempt_id, 'read_file',
            {'path': '/tmp/test.txt'}, 'file contents'
        )

        actions = shots_on_goal.get_actions(self.db, self.attempt_id)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]['id'], action_id)
        self.assertEqual(actions[0]['tool_name'], 'read_file')
        self.assertIn('path', actions[0]['parameters'])

    def test_multiple_actions(self):
        """Test recording multiple actions"""
        shots_on_goal.record_action(
            self.db, self.attempt_id, 'read_file',
            {'path': 'BUILD'}, 'contents'
        )
        shots_on_goal.record_action(
            self.db, self.attempt_id, 'write_file',
            {'path': 'BUILD', 'content': 'new'}, 'success'
        )
        shots_on_goal.record_action(
            self.db, self.attempt_id, 'run_command',
            {'cmd': 'bazel build'}, 'output'
        )

        actions = shots_on_goal.get_actions(self.db, self.attempt_id)
        self.assertEqual(len(actions), 3)


class TestSessionManagement(unittest.TestCase):
    """Test session creation and management"""

    def setUp(self):
        """Create temporary directory for test sessions"""
        self.temp_dir = tempfile.mkdtemp()

        # Create a temporary git repo
        self.repo_dir = Path(self.temp_dir) / "test-repo"
        self.repo_dir.mkdir()
        subprocess.run(['git', 'init'], cwd=self.repo_dir, check=True,
                      capture_output=True)

        # Monkey-patch sessions directory
        self.sessions_dir = Path(self.temp_dir) / "sessions"
        self.sessions_dir.mkdir()
        self.get_sessions_dir_patch = patch(
            'shots_on_goal.get_sessions_dir',
            return_value=self.sessions_dir
        )
        self.get_sessions_dir_patch.start()

    def tearDown(self):
        """Clean up temporary files"""
        self.get_sessions_dir_patch.stop()
        shutil.rmtree(self.temp_dir)

    def test_create_session(self):
        """Test creating a new session"""
        session_dir, db = shots_on_goal.create_session(
            "Test goal", str(self.repo_dir)
        )

        # Check session directory exists
        self.assertTrue(session_dir.exists())
        self.assertTrue((session_dir / "goals.db").exists())

        # Check session record
        session = shots_on_goal.get_session_record(db)
        self.assertEqual(session['description'], "Test goal")
        self.assertEqual(session['status'], 'active')

        db.close()

    def test_load_session(self):
        """Test loading an existing session"""
        # Create session
        session_dir, db = shots_on_goal.create_session(
            "Test goal", str(self.repo_dir)
        )
        db.close()

        # Load it back
        loaded_dir, loaded_db, session_record = shots_on_goal.load_session(
            str(session_dir)
        )

        self.assertEqual(str(loaded_dir), str(session_dir))
        self.assertEqual(session_record['description'], "Test goal")

        loaded_db.close()


class TestGitManager(unittest.TestCase):
    """Test git operations and worktree management"""

    def setUp(self):
        """Create temporary git repository for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_dir = Path(self.temp_dir) / "test-repo"
        self.repo_dir.mkdir()

        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_dir, check=True,
                      capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'],
                      cwd=self.repo_dir, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'],
                      cwd=self.repo_dir, check=True, capture_output=True)

        # Create initial commit
        (self.repo_dir / "README.md").write_text("# Test Repo\n")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_dir, check=True,
                      capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'],
                      cwd=self.repo_dir, check=True, capture_output=True)

        self.git_mgr = shots_on_goal.GitManager(str(self.repo_dir))

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)

    def test_create_session_branch(self):
        """Test creating a session branch"""
        branch_name = self.git_mgr.create_session_branch('20251016-100000')
        self.assertEqual(branch_name, 'session-20251016-100000')

        # Verify branch exists
        result = subprocess.run(
            ['git', 'branch', '--list', branch_name],
            cwd=self.repo_dir,
            capture_output=True,
            text=True
        )
        self.assertIn(branch_name, result.stdout)

    def test_create_goal_branch(self):
        """Test creating a goal branch"""
        # Create session branch first
        session_branch = self.git_mgr.create_session_branch('20251016-100000')

        # Create goal branch from session branch
        goal_branch = self.git_mgr.create_goal_branch(1, session_branch)
        self.assertEqual(goal_branch, 'goal-1')

        # Verify branch exists
        result = subprocess.run(
            ['git', 'branch', '--list', goal_branch],
            cwd=self.repo_dir,
            capture_output=True,
            text=True
        )
        self.assertIn(goal_branch, result.stdout)

    def test_create_worktree_for_attempt(self):
        """Test creating a worktree for an attempt"""
        session_branch = self.git_mgr.create_session_branch('20251016-100000')

        worktree_path, branch_name, commit_sha = \
            self.git_mgr.create_worktree_for_attempt(1, 1, session_branch)

        # Check return values
        self.assertIn('goal-1-attempt-1', worktree_path)
        self.assertEqual(branch_name, 'goal-1-attempt-1')
        self.assertTrue(len(commit_sha) == 40)  # SHA is 40 chars

        # Verify worktree exists
        worktree_path_obj = Path(worktree_path)
        self.assertTrue(worktree_path_obj.exists())
        self.assertTrue((worktree_path_obj / "README.md").exists())

    def test_get_current_commit_sha(self):
        """Test getting current commit SHA"""
        commit_sha = self.git_mgr.get_current_commit_sha()
        self.assertTrue(len(commit_sha) == 40)

    def test_merge_branch(self):
        """Test merging branches"""
        # Create two branches
        session_branch = self.git_mgr.create_session_branch('20251016-100000')
        goal_branch = self.git_mgr.create_goal_branch(1, session_branch)

        # Make a change in goal branch
        (self.repo_dir / "test.txt").write_text("test content")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_dir, check=True,
                      capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add test file'],
                      cwd=self.repo_dir, check=True, capture_output=True)

        # Merge goal branch back to session branch
        self.git_mgr.merge_branch(goal_branch, session_branch)

        # Verify file exists in session branch
        subprocess.run(['git', 'checkout', session_branch],
                      cwd=self.repo_dir, check=True, capture_output=True)
        self.assertTrue((self.repo_dir / "test.txt").exists())

    def test_remove_worktree(self):
        """Test removing a worktree"""
        session_branch = self.git_mgr.create_session_branch('20251016-100000')
        worktree_path, _, _ = self.git_mgr.create_worktree_for_attempt(
            1, 1, session_branch
        )

        # Worktree should exist
        self.assertTrue(Path(worktree_path).exists())

        # Remove it
        self.git_mgr.remove_worktree(worktree_path)

        # Worktree should no longer exist
        self.assertFalse(Path(worktree_path).exists())

    def test_list_worktrees(self):
        """Test listing worktrees"""
        session_branch = self.git_mgr.create_session_branch('20251016-100000')

        # Create a couple worktrees
        self.git_mgr.create_worktree_for_attempt(1, 1, session_branch)
        self.git_mgr.create_worktree_for_attempt(1, 2, session_branch)

        worktrees = self.git_mgr.list_worktrees()

        # Should have exactly 3: main repo + 2 worktrees
        self.assertEqual(len(worktrees), 3,
                        f"Expected 3 worktrees but got {len(worktrees)}: {worktrees}")

        # Find our worktrees
        worktree_branches = [w.get('branch', '') for w in worktrees]
        self.assertIn('refs/heads/goal-1-attempt-1', worktree_branches)
        self.assertIn('refs/heads/goal-1-attempt-2', worktree_branches)


class TestContainerManager(unittest.TestCase):
    """Test container management"""

    def setUp(self):
        """Check if container runtime is available"""
        self.temp_dir = tempfile.mkdtemp()

        # Try to detect container runtime
        self.runtime = None
        for cmd in ['container', 'docker']:
            try:
                subprocess.run([cmd, '--version'], check=True,
                             capture_output=True)
                self.runtime = cmd
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if not self.runtime:
            self.skipTest("No container runtime (container or docker) available")

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)

    def test_container_start_stop(self):
        """Test starting and stopping a container"""
        # Create a test workspace
        workspace = Path(self.temp_dir) / "workspace"
        workspace.mkdir()
        (workspace / "test.txt").write_text("hello")

        container = shots_on_goal.ContainerManager(
            image="ubuntu:22.04",
            runtime=self.runtime
        )

        # Start container
        container_id = container.start(str(workspace))
        self.assertIsNotNone(container_id)
        self.assertTrue(len(container_id) > 0)

        # Stop container
        container.stop()
        self.assertIsNone(container.container_id)

    def test_container_exec(self):
        """Test executing commands in container"""
        workspace = Path(self.temp_dir) / "workspace"
        workspace.mkdir()
        (workspace / "test.txt").write_text("hello world")

        container = shots_on_goal.ContainerManager(
            image="ubuntu:22.04",
            runtime=self.runtime
        )

        try:
            container.start(str(workspace))

            # Execute command to read file
            result = container.exec("cat test.txt")
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "hello world")

            # Execute command that fails
            result = container.exec("cat nonexistent.txt")
            self.assertNotEqual(result.returncode, 0)

        finally:
            container.stop()

    def test_container_context_manager(self):
        """Test using container as context manager"""
        workspace = Path(self.temp_dir) / "workspace"
        workspace.mkdir()

        with shots_on_goal.ContainerManager(
            image="ubuntu:22.04",
            runtime=self.runtime
        ) as container:
            container.start(str(workspace))
            self.assertIsNotNone(container.container_id)

            # Execute a command
            result = container.exec("pwd")
            self.assertEqual(result.returncode, 0)
            self.assertIn("workspace", result.stdout)

        # Container should be stopped after exiting context
        self.assertIsNone(container.container_id)

    def test_exec_without_start_raises_error(self):
        """Test that exec raises error if container not started"""
        container = shots_on_goal.ContainerManager(runtime=self.runtime)

        with self.assertRaises(RuntimeError):
            container.exec("echo hello")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionRecord))
    suite.addTests(loader.loadTestsFromTestCase(TestGoals))
    suite.addTests(loader.loadTestsFromTestCase(TestAttempts))
    suite.addTests(loader.loadTestsFromTestCase(TestActions))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestGitManager))
    suite.addTests(loader.loadTestsFromTestCase(TestContainerManager))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

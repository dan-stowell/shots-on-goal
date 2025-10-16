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

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

import unittest
from src import error_logger
from src import config

class TestErrorLoggerAndHostAuth(unittest.TestCase):
    def setUp(self):
        error_logger.clear_all_errors()

    def tearDown(self):
        error_logger.clear_all_errors()

    def test_record_and_get_errors(self):
        """Test recording an error and retrieving recent error logs."""
        self.assertEqual(len(error_logger.get_recent_errors()), 0)
        self.assertEqual(error_logger.get_unacknowledged_count(), 0)

        err = error_logger.record_error(
            service="Pinecone",
            user_message="Connection timeout during query",
            technical_details="Traceback: simulated timeout",
        )
        self.assertEqual(err["service"], "Pinecone")
        self.assertEqual(err["message"], "Connection timeout during query")
        self.assertFalse(err["acknowledged"])

        self.assertEqual(error_logger.get_unacknowledged_count(), 1)
        recent = error_logger.get_recent_errors()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["service"], "Pinecone")

    def test_acknowledge_and_clear_errors(self):
        """Test acknowledging and clearing error logs."""
        error_logger.record_error("Gemini", "Quota limit reached")
        self.assertEqual(error_logger.get_unacknowledged_count(), 1)

        error_logger.acknowledge_all_errors()
        self.assertEqual(error_logger.get_unacknowledged_count(), 0)
        self.assertEqual(len(error_logger.get_recent_errors()), 1)

        error_logger.clear_all_errors()
        self.assertEqual(len(error_logger.get_recent_errors()), 0)

    def test_host_admin_pin(self):
        """Verify host admin PIN configuration defaults."""
        self.assertTrue(hasattr(config, "HOST_ADMIN_PIN"))
        self.assertTrue(len(config.HOST_ADMIN_PIN) > 0)

if __name__ == "__main__":
    unittest.main()


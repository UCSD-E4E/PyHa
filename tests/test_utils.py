import io
import unittest
import unittest.mock

from PyHa.utils import check_verbose


class CheckVerbose(unittest.TestCase):
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def assert_stdout(self, message, expected_message, verbose, mock_stdout):
        check_verbose(message, verbose)
        self.assertEqual(mock_stdout.getvalue(), expected_message)

    def test_verbose(self):
        ERROR_MESSAGE = "Hello World"
        self.assert_stdout(ERROR_MESSAGE, ERROR_MESSAGE + "\n", True)
        self.assert_stdout(ERROR_MESSAGE, "", False)

        self.assert_stdout(ERROR_MESSAGE, ERROR_MESSAGE + "\n", {"verbose": True})
        self.assert_stdout(ERROR_MESSAGE, "", {"verbose": False})


if __name__ == "__main__":
    unittest.main()

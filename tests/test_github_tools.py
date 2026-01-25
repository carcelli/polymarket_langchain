import unittest
from unittest.mock import MagicMock, patch
import os

from polymarket_agents.tools.github_tools import (
    _get_github_toolkit,
    _get_issues_impl,
    _get_issue_impl,
    _create_issue_comment_impl,
)


class TestGitHubTools(unittest.TestCase):

    @patch("polymarket_agents.tools.github_tools.GitHubToolkit")
    @patch("polymarket_agents.tools.github_tools.GitHubAPIWrapper")
    @patch.dict(
        os.environ,
        {
            "GITHUB_APP_ID": "123456",
            "GITHUB_APP_PRIVATE_KEY": "fake_key",
            "GITHUB_REPOSITORY": "fake/repo",
        },
    )
    def test_get_github_toolkit_success(self, mock_wrapper, mock_toolkit):
        toolkit = _get_github_toolkit()
        self.assertIsNotNone(toolkit)
        mock_wrapper.assert_called_once()
        mock_toolkit.from_github_api_wrapper.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_get_github_toolkit_missing_env(self):
        # Ensure specific env vars are missing
        if "GITHUB_APP_ID" in os.environ:
            del os.environ["GITHUB_APP_ID"]

        toolkit = _get_github_toolkit()
        self.assertIsNone(toolkit)

    @patch("polymarket_agents.tools.github_tools._get_github_toolkit")
    def test_get_issues_impl(self, mock_get_toolkit):
        mock_tool = MagicMock()
        mock_tool.name = "Get Issues"
        mock_tool.invoke.return_value = "Issue list"

        mock_toolkit = MagicMock()
        mock_toolkit.get_tools.return_value = [mock_tool]
        mock_get_toolkit.return_value = mock_toolkit

        result = _get_issues_impl()
        self.assertEqual(result, "Issue list")
        mock_tool.invoke.assert_called_once()

    @patch("polymarket_agents.tools.github_tools._get_github_toolkit")
    def test_get_issue_impl(self, mock_get_toolkit):
        mock_tool = MagicMock()
        mock_tool.name = "Get Issue"
        mock_tool.invoke.return_value = "Issue details"

        mock_toolkit = MagicMock()
        mock_toolkit.get_tools.return_value = [mock_tool]
        mock_get_toolkit.return_value = mock_toolkit

        result = _get_issue_impl(1)
        self.assertEqual(result, "Issue details")
        mock_tool.invoke.assert_called_once_with({"issue_number": 1})

    @patch("polymarket_agents.tools.github_tools._get_github_toolkit")
    def test_create_issue_comment_impl(self, mock_get_toolkit):
        mock_tool = MagicMock()
        mock_tool.name = "Comment on Issue"
        mock_tool.invoke.return_value = "Comment created"

        mock_toolkit = MagicMock()
        mock_toolkit.get_tools.return_value = [mock_tool]
        mock_get_toolkit.return_value = mock_toolkit

        result = _create_issue_comment_impl(1, "Test comment")
        self.assertEqual(result, "Comment created")
        mock_tool.invoke.assert_called_once_with(
            {"issue_number": 1, "body": "Test comment"}
        )


if __name__ == "__main__":
    unittest.main()

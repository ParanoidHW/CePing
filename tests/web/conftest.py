"""Pytest configuration for web frontend tests."""


def pytest_configure(config):
    """Mark web tests."""
    config.addinivalue_line("markers", "web: mark test as web frontend test (requires playwright)")
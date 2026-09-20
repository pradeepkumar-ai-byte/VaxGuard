import asyncio
import pytest
from vaxguard.db.session import init_db


@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    """Ensure database schema is bootstraped before any test runs in any environment."""
    asyncio.run(init_db())

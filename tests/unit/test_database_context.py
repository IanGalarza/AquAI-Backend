"""Pruebas de configuración, motor diferido y transacciones sin acceder a la BD."""

import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from aquai.bootstrap import initialize_application
from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.db_context import DbContext
from aquai.infrastructure.providers.database import get_db_session


def settings(**overrides: object) -> DatabaseSettings:
    return DatabaseSettings(
        _env_file=None,
        database_url="postgresql+psycopg://aquai:private_password@localhost/aquai",
        **overrides,
    )


class DatabaseSettingsTests(unittest.TestCase):
    def test_settings_load_environment_and_ignore_compose_variables(self) -> None:
        with patch.dict("os.environ", {
            "DATABASE_URL": "postgresql://aquai:secret@localhost/aquai",
            "POSTGRES_PASSWORD": "compose_secret",
        }, clear=True):
            loaded = DatabaseSettings(_env_file=None)
        self.assertEqual(loaded.sqlalchemy_url.drivername, "postgresql+psycopg")
        self.assertNotIn("secret", repr(loaded))

    def test_invalid_settings_fail_without_exposing_credentials(self) -> None:
        for url in (
            "sqlite:///aquai.db", "postgresql+asyncpg://user:private_password@localhost/aquai",
            "postgresql+psycopg://user:private_password@localhost", "",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValidationError) as raised:
                    DatabaseSettings(_env_file=None, database_url=url)
                self.assertNotIn("private_password", str(raised.exception))

    def test_missing_database_url_does_not_silently_choose_a_database(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValidationError):
                DatabaseSettings(_env_file=None)


class DbContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_application()

    def setUp(self) -> None:
        self.engine = MagicMock()
        self.session = MagicMock()
        self.factory = MagicMock(return_value=self.session)
        self.create_engine = patch(
            "aquai.infrastructure.persistence.db_context.create_engine", return_value=self.engine
        ).start()
        self.create_factory = patch(
            "aquai.infrastructure.persistence.db_context.sessionmaker", return_value=self.factory
        ).start()
        self.addCleanup(patch.stopall)
        self.context = DbContext(settings())

    def test_engine_is_lazy_and_metadata_never_connects(self) -> None:
        self.create_engine.assert_not_called()
        self.assertEqual(len(self.context.metadata.tables), 9)
        self.create_engine.assert_not_called()
        self.assertIs(self.context.engine, self.engine)
        self.assertIs(self.context.engine, self.engine)
        self.create_engine.assert_called_once_with(
            self.context._settings.sqlalchemy_url
        )
        self.create_factory.assert_called_once_with(bind=self.engine, expire_on_commit=False)
        self.engine.connect.assert_not_called()

    def test_startup_registers_mappings_without_database_settings_or_engine(self) -> None:
        # Un intérprete nuevo reproduce el arranque sin mapeos previos ni variables
        # de configuración. Agregar una entidad a una Session sin motor no accede a la BD.
        code = """
from unittest.mock import patch
from sqlalchemy.orm import Session
from aquai.bootstrap import initialize_application
from aquai.domain.models import Conversation
from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.db_context import DbContext
from aquai.infrastructure.persistence.mapping import mapper_registry, metadata
with patch.dict('os.environ', {}, clear=True), patch(
    'aquai.infrastructure.persistence.db_context.create_engine'
) as create_engine:
    assert not hasattr(Conversation, '__mapper__')
    initialize_application()
    first_mapper = Conversation.__mapper__
    initialize_application()
    assert Conversation.__mapper__ is first_mapper
    assert len(list(mapper_registry.mappers)) == 9
    assert len(metadata.tables) == 9
    conversation = Conversation(title='startup')
    with Session() as session:
        session.add(conversation)
        assert conversation in session.new
    create_engine.assert_not_called()
    context = DbContext(DatabaseSettings(_env_file=None, database_url='postgresql+psycopg://localhost/aquai'))
    assert context._engine is None
    create_engine.assert_not_called()
"""
        result = subprocess.run(
            [sys.executable, "-B", "-W", "error", "-c", code],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_context_does_not_register_mappings(self) -> None:
        code = """
from unittest.mock import MagicMock, patch
from aquai.domain.models import Conversation
from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.db_context import DbContext
with patch('aquai.infrastructure.persistence.db_context.create_engine', return_value=MagicMock()), patch(
    'aquai.infrastructure.persistence.db_context.sessionmaker', return_value=MagicMock()
):
    context = DbContext(DatabaseSettings(_env_file=None, database_url='postgresql+psycopg://localhost/aquai'))
    assert not hasattr(Conversation, '__mapper__')
    assert len(context.metadata.tables) == 0
    assert not hasattr(Conversation, '__mapper__')
    with context.session():
        assert not hasattr(Conversation, '__mapper__')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-W", "error", "-c", code],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_success_requires_explicit_commit_and_always_closes(self) -> None:
        with self.context.session() as session:
            self.assertIs(session, self.session)
        self.session.commit.assert_not_called()
        self.session.rollback.assert_not_called()
        self.session.close.assert_called_once_with()

    def test_caller_can_commit_explicitly(self) -> None:
        with self.context.session() as session:
            session.commit()
        self.session.commit.assert_called_once_with()
        self.session.rollback.assert_not_called()
        self.session.close.assert_called_once_with()

    def test_application_error_rolls_back_and_always_closes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "falló la aplicación"):
            with self.context.session():
                raise RuntimeError("falló la aplicación")
        self.session.commit.assert_not_called()
        self.session.rollback.assert_called_once_with()
        self.session.close.assert_called_once_with()

    def test_commit_failure_rolls_back_and_always_closes(self) -> None:
        self.session.commit.side_effect = RuntimeError("falló el commit")
        with self.assertRaisesRegex(RuntimeError, "falló el commit"):
            with self.context.session() as session:
                session.commit()
        self.session.commit.assert_called_once_with()
        self.session.rollback.assert_called_once_with()
        self.session.close.assert_called_once_with()

    def test_fastapi_dependency_manages_success_and_failure(self) -> None:
        with patch("aquai.infrastructure.providers.database.get_db_context", return_value=self.context):
            dependency = get_db_session()
            self.assertIs(next(dependency), self.session)
            with self.assertRaises(StopIteration):
                next(dependency)
            self.session.commit.assert_not_called()
            self.session.rollback.assert_not_called()
            self.session.close.assert_called_once_with()
            self.session.reset_mock()
            dependency = get_db_session()
            next(dependency)
            with self.assertRaisesRegex(RuntimeError, "falló la petición"):
                dependency.throw(RuntimeError("falló la petición"))
            self.session.commit.assert_not_called()
            self.session.rollback.assert_called_once_with()
            self.session.close.assert_called_once_with()

    def test_dispose_releases_engine_and_allows_reuse(self) -> None:
        self.context.dispose()
        self.create_engine.assert_not_called()
        _ = self.context.engine
        self.context.dispose()
        self.engine.dispose.assert_called_once_with()
        _ = self.context.engine
        self.assertEqual(self.create_engine.call_count, 2)


if __name__ == "__main__":
    unittest.main()

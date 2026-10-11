import hashlib
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

Base = declarative_base()
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    if settings.DATABASE_URL.startswith("sqlite://"):
        db_path = settings.DATABASE_URL.replace("sqlite:///", "", 1)
        if db_path.startswith("/"):
            db_path = db_path[1:]
        path = Path(db_path)
        if path.parent.exists() is False:
            path.parent.mkdir(parents=True, exist_ok=True)

    from .models import ChatMessage, ChunkRecord, DocumentRecord, PageRecord  # noqa: F401

    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    migrations = {
        "documents": {
            "sha256": "VARCHAR(64)",
        },
        "pages": {
            "extraction_error": "TEXT",
        },
        "chunks": {
            "embedding_data": "TEXT",
            "embedding_model": "VARCHAR(255)",
        },
    }
    with engine.begin() as connection:
        for table_name, columns in migrations.items():
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name not in existing:
                    connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
        connection.execute(
            text("CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_sha256 ON documents (sha256)")
        )
        upload_root = Path(settings.UPLOAD_DIR).resolve()
        rows = connection.execute(
            text("SELECT id, storage_path FROM documents WHERE sha256 IS NULL")
        ).all()
        known_hashes = {
            row[0]
            for row in connection.execute(text("SELECT sha256 FROM documents WHERE sha256 IS NOT NULL")).all()
        }
        for document_id, storage_path in rows:
            path = Path(storage_path).resolve()
            try:
                path.relative_to(upload_root)
            except ValueError:
                continue
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            file_hash = digest.hexdigest()
            if file_hash not in known_hashes:
                connection.execute(
                    text("UPDATE documents SET sha256 = :sha256 WHERE id = :document_id"),
                    {"sha256": file_hash, "document_id": document_id},
                )
                known_hashes.add(file_hash)

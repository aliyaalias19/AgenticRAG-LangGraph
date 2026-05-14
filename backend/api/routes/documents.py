"""Document management routes: ingest, list, delete. Tenant-scoped."""

import logging
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from core.audit import audit_event
from core.auth import TenantContext, get_tenant_context
from core.config import get_settings
from core.exceptions import IngestionError
from ingestion.pipeline import delete_document, ingest_pdf, list_ingested_files
from models.responses import DeleteResponse, DocumentsResponse, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()


@router.post("", response_model=IngestResponse, summary="Ingest a PDF")
async def ingest_document(
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Upload and ingest a single PDF into the caller's tenant corpus.

    Tenants are isolated end-to-end: a PDF ingested under tenant A is
    invisible to retrieval under tenant B even if the same filename and
    SHA-256 content are uploaded under both.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Tenant-scoped upload directory prevents one tenant from overwriting
    # another's PDF by uploading the same filename. The vector store
    # already partitions by tenant_id; this just keeps the on-disk
    # filesystem layout coherent with the data model.
    save_path = Path(settings.data_dir) / tenant.tenant_id / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    save_path.write_bytes(content)
    logger.info("Saved upload", extra={
        "path": str(save_path), "tenant_id": tenant.tenant_id,
    })

    try:
        result = ingest_pdf(str(save_path), settings, tenant_id=tenant.tenant_id)
        audit_event(
            "ingest",
            tenant_id=tenant.tenant_id,
            actor_fingerprint=tenant.api_key_fingerprint,
            detail={"file": file.filename, "status": result.get("status"),
                    "chunks": result.get("chunks", 0)},
        )
        return IngestResponse(**result)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed", extra={"file": file.filename})
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.post("/bulk", summary="Ingest all PDFs in the tenant's data directory")
async def ingest_directory(tenant: TenantContext = Depends(get_tenant_context)):
    """Batch-ingest every PDF in the tenant-scoped data directory.

    Looks under ``${DATA_DIR}/${tenant_id}/`` first, falling back to the
    flat ``${DATA_DIR}`` for the demo tenant so the default single-
    tenant workflow keeps working without operators rearranging files.
    """
    tenant_dir = Path(settings.data_dir) / tenant.tenant_id
    fallback_dir = Path(settings.data_dir)
    data_path = tenant_dir if tenant_dir.exists() else fallback_dir

    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"DATA_DIR not found: {data_path}")

    pdfs = sorted(data_path.glob("*.pdf"))
    if not pdfs:
        return {"message": "No PDFs found.", "results": []}

    results = []
    for pdf in pdfs:
        try:
            result = ingest_pdf(str(pdf), settings, tenant_id=tenant.tenant_id)
            results.append(result)
        except Exception as exc:
            logger.exception("Failed to ingest", extra={"file": pdf.name})
            results.append({"status": "error", "file": pdf.name, "error": str(exc)})

    audit_event(
        "ingest_bulk",
        tenant_id=tenant.tenant_id,
        actor_fingerprint=tenant.api_key_fingerprint,
        detail={"file_count": len(pdfs)},
    )
    return {"total": len(pdfs), "results": results}


@router.get("", response_model=DocumentsResponse, summary="List ingested documents")
async def list_documents(tenant: TenantContext = Depends(get_tenant_context)):
    files = list_ingested_files(settings, tenant_id=tenant.tenant_id)
    return DocumentsResponse(documents=files, count=len(files))


@router.delete("/{filename}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document_endpoint(
    filename: str,
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Remove all chunks for the given document, scoped to the caller's tenant."""
    filename = unquote(filename)
    try:
        result = delete_document(filename, settings, tenant_id=tenant.tenant_id)
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
        audit_event(
            "delete",
            tenant_id=tenant.tenant_id,
            actor_fingerprint=tenant.api_key_fingerprint,
            detail={"file": filename, "chunks_deleted": result.get("chunks_deleted", 0)},
        )
        return DeleteResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Delete failed", extra={"file": filename})
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")

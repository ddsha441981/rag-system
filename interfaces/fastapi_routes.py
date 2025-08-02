from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import tempfile
import os
from .schemas import QueryRequest, QueryResponse, DocumentResponse, SystemStats
from core.rag_system import EnhancedRAGSystem
from app.config import settings

router = APIRouter()

rag_system = None


def get_rag_system():
    global rag_system
    if rag_system is None:
        rag_system = EnhancedRAGSystem(
            groq_api_key=settings.groq_api_key,
            collection_name=settings.collection_name,
            redis_host=settings.redis_host,
            redis_port=settings.redis_port
        )
    return rag_system


@router.post("/upload", response_model=List[DocumentResponse])
async def upload_documents(files: List[UploadFile] = File(...),
                           system: EnhancedRAGSystem = Depends(get_rag_system)):

    results = []

    for file in files:
        try:
            # Validate file type
            file_ext = file.filename.split('.')[-1].lower()
            if file_ext not in settings.allowed_file_types:
                results.append(DocumentResponse(
                    filename=file.filename,
                    chunks_processed=0,
                    success=False,
                    error=f"File type '{file_ext}' not supported"
                ))
                continue

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                content = await file.read()

                # Check file size
                if len(content) > settings.max_upload_size_mb * 1024 * 1024:
                    results.append(DocumentResponse(
                        filename=file.filename,
                        chunks_processed=0,
                        success=False,
                        error=f"File size exceeds {settings.max_upload_size_mb}MB limit"
                    ))
                    continue

                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            # Process document
            chunks = system.process_document(tmp_file_path)

            results.append(DocumentResponse(
                filename=file.filename,
                chunks_processed=chunks,
                success=True
            ))

        except Exception as e:
            results.append(DocumentResponse(
                filename=file.filename,
                chunks_processed=0,
                success=False,
                error=str(e)
            ))
        finally:
            # Clean up temporary file
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

    return results


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest,
                          system: EnhancedRAGSystem = Depends(get_rag_system)):

    try:

        conv_context = ""
        if request.session_id:
            conv_context = system.conversation_memory.get_conversation_context(request.session_id)

        # Process query
        result = system.query(
            request.query,
            k=request.k,
            conversation_context=conv_context,
            search_mode=request.search_mode,
            session_id=request.session_id
        )

        if result['success']:
            return QueryResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Query processing failed'))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_cache(system: EnhancedRAGSystem = Depends(get_rag_system)):

    try:
        system.query_cache.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=SystemStats)
async def get_stats(system: EnhancedRAGSystem = Depends(get_rag_system)):

    try:
        stats = system.get_stats()
        return SystemStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str, system: EnhancedRAGSystem = Depends(get_rag_system)):

    try:
        system.conversation_memory.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

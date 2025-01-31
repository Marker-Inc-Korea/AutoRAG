from celery.result import AsyncResult
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_task_info(task_id: str) -> Dict[str, Any]:
    """Get detailed information about a task"""
    result = AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": result.status,
        "success": result.successful() if result.ready() else None,
    }
    
    if result.ready():
        if result.successful():
            response["result"] = result.get()
        else:
            response["error"] = str(result.result)
            
    if hasattr(result, 'info'):
        response["progress"] = result.info
        
    return response

def cleanup_old_tasks(days: int = 7):
    """Clean up tasks older than specified days"""
    try:
        # Implementation depends on your storage method
        pass
    except Exception as e:
        logger.error(f"Error cleaning up old tasks: {e}") 
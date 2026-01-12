import logging
from typing import Any, Dict

logger = logging.getLogger("voice_agent_v4.mcp_tools.crm_call_management")

def end_call(reason: str = "conversation_ended") -> Dict[str, Any]:
    """
    End the current call when the conversation has reached a natural conclusion.
    
    This function signals the voice agent to terminate the connection immediately 
    after the current response is finished.
    """
    try:
        # Access the shared context var from the adapter (optional in text-only chat)
        try:
            from voice_agent.providers.openai_adapter import _get_call_context  # type: ignore
        except ImportError:
            logger.warning("Call context adapter missing; ending conversation best-effort")
            return {
                "success": True,
                "status": "ending_call",
                "message": "Ending conversation (no call context available).",
            }

        ctx = _get_call_context()
        if ctx is not None:
            ctx["should_hangup"] = True
            ctx["hangup_reason"] = reason
            logger.info("Signal to end call received: reason=%s", reason)
            return {"success": True, "status": "ending_call", "message": "Call will end after this response."}

        logger.warning("Attempted to end call but no active call context found")
        return {"success": False, "error": "No active call context found"}
            
    except Exception as e:
        logger.error("Failed to signal end call: %s", e)
        return {"success": False, "error": str(e)}

TOOL_DEFINITION = {
    "name": "end_call",
    "description": "End the call when the conversation is finished or the user requests to hang up.",
    "parameters": {
        "reason": {
            "type": "string",
            "description": "Reason for ending the call (e.g., 'user_request', 'task_completed')",
            "default": "conversation_ended"
        }
    },
    "required": [],
    "handler": end_call
}

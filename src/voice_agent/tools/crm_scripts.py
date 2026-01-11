"""
CRM Call Scripts Tools

Provides tools for accessing call scripts and objection handlers in Supabase.
These help the agent follow best practices during calls.
"""

import logging
from typing import Any, Dict, Optional

from voice_agent.utils.db import get_supabase_client


logger = logging.getLogger(__name__)


# Tool implementations

def get_call_script(call_type: str, industry: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the call script for a specific call type and optionally industry.
    
    Provides greeting, qualification questions, objection handlers, and call goals.
    
    Args:
        call_type: Type of call (outbound_cold_call, inbound_support, follow_up, demo, etc.)
        industry: Industry vertical to filter by (optional)
        
    Returns:
        Call script details
    """
    try:
        supabase = get_supabase_client()
        
        query = supabase.table("call_scripts").select(
            "id, name, call_type, industry, greeting, qualification_questions, objection_handlers, call_goals, closing_statements, active, created_at"
        ).eq("call_type", call_type).eq("active", True)
        
        if industry:
            query = query.eq("industry", industry)
            
        # Order by created_at desc
        response = query.order("created_at", desc=True).limit(1).execute()
        
        if response.data:
            row = response.data[0]
            
            # Supabase returns JSONB columns as python objects (list/dict) or None
            qualification_questions = row.get("qualification_questions") or []
            objection_handlers = row.get("objection_handlers") or {}
            call_goals = row.get("call_goals") or []
            
            return {
                "found": True,
                "script_id": row["id"],
                "name": row["name"],
                "call_type": row["call_type"],
                "industry": row["industry"],
                "greeting": row["greeting"],
                "qualification_questions": qualification_questions,
                "objection_handlers": objection_handlers,
                "call_goals": call_goals,
                "closing_statements": row["closing_statements"],
                "created_at": row["created_at"],
            }
        else:
            message = f"No active script found for call_type: {call_type}"
            if industry:
                message += f", industry: {industry}"
            return {
                "found": False,
                "message": message,
                "call_type": call_type,
                "industry": industry,
            }
        
    except Exception as e:
        logger.error(f"Error fetching call script: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "call_type": call_type,
            "industry": industry,
        }


def get_all_call_scripts(active_only: bool = True) -> Dict[str, Any]:
    """
    Get all available call scripts.
    
    Args:
        active_only: Only return active scripts (default True)
        
    Returns:
        List of all call scripts
    """
    try:
        supabase = get_supabase_client()
        
        query = supabase.table("call_scripts").select(
            "id, name, call_type, industry, greeting, qualification_questions, objection_handlers, call_goals, closing_statements, active, created_at"
        )
        
        if active_only:
            query = query.eq("active", True)
            
        response = query.order("call_type").order("created_at", desc=True).execute()
        
        scripts = []
        for row in response.data:
            qualification_questions = row.get("qualification_questions") or []
            objection_handlers = row.get("objection_handlers") or {}
            call_goals = row.get("call_goals") or []
            
            scripts.append({
                "script_id": row["id"],
                "name": row["name"],
                "call_type": row["call_type"],
                "industry": row["industry"],
                "greeting": row["greeting"],
                "qualification_questions": qualification_questions,
                "objection_handlers": objection_handlers,
                "call_goals": call_goals,
                "closing_statements": row["closing_statements"],
                "active": row["active"],
                "created_at": row["created_at"],
            })
        
        return {
            "success": True,
            "script_count": len(scripts),
            "scripts": scripts,
        }
        
    except Exception as e:
        logger.error(f"Error fetching all call scripts: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def get_objection_handler(call_type: str, objection_key: str) -> Dict[str, Any]:
    """
    Get a specific objection handler response.
    
    Args:
        call_type: Type of call to get script for
        objection_key: Key identifying the objection (e.g., "muito_caro", "nao_interessado")
        
    Returns:
        Suggested response to the objection
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("call_scripts").select(
            "id, name, objection_handlers"
        ).eq("call_type", call_type).eq("active", True).order("created_at", desc=True).limit(1).execute()
        
        if response.data:
            row = response.data[0]
            objection_handlers = row.get("objection_handlers") or {}
            
            if objection_key in objection_handlers:
                return {
                    "found": True,
                    "script_id": row["id"],
                    "script_name": row["name"],
                    "call_type": call_type,
                    "objection_key": objection_key,
                    "response": objection_handlers[objection_key],
                }
            else:
                return {
                    "found": False,
                    "message": f"No handler found for objection: {objection_key}",
                    "call_type": call_type,
                    "objection_key": objection_key,
                    "available_objections": list(objection_handlers.keys()),
                }
        else:
            return {
                "found": False,
                "message": f"No active script found for call_type: {call_type}",
                "call_type": call_type,
                "objection_key": objection_key,
            }
        
    except Exception as e:
        logger.error(f"Error fetching objection handler: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "call_type": call_type,
            "objection_key": objection_key,
        }


def list_objections(call_type: str) -> Dict[str, Any]:
    """
    List all available objection handlers for a call type.
    
    Args:
        call_type: Type of call to get objections for
        
    Returns:
        Dictionary of objection keys and responses
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("call_scripts").select(
            "id, name, objection_handlers"
        ).eq("call_type", call_type).eq("active", True).order("created_at", desc=True).limit(1).execute()
        
        if response.data:
            row = response.data[0]
            objection_handlers = row.get("objection_handlers") or {}
            
            return {
                "found": True,
                "script_id": row["id"],
                "script_name": row["name"],
                "call_type": call_type,
                "objection_count": len(objection_handlers),
                "objections": objection_handlers,
            }
        else:
            return {
                "found": False,
                "message": f"No active script found for call_type: {call_type}",
                "call_type": call_type,
            }
        
    except Exception as e:
        logger.error(f"Error listing objections: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "call_type": call_type,
        }


TOOL_DEFINITIONS = [
    {
        "name": "get_call_script",
        "description": (
            "Retrieve the full script for a call type, including greeting, "
            "questions, objections, and goals."
        ),
        "parameters": {
            "call_type": {
                "type": "string",
                "description": "Call type (outbound_cold_call, inbound_support, etc.)",
            },
        },
        "required": ["call_type"],
        "handler": get_call_script,
    },
    {
        "name": "get_all_call_scripts",
        "description": (
            "List all configured call scripts to understand which playbooks are "
            "available."
        ),
        "parameters": {
            "active_only": {
                "type": "boolean",
                "description": "Only return active scripts (default True)",
                "default": True,
            },
        },
        "handler": get_all_call_scripts,
    },
    {
        "name": "get_objection_handler",
        "description": (
            "Fetch the suggested response for a specific objection raised during "
            "the call."
        ),
        "parameters": {
            "call_type": {
                "type": "string",
                "description": "Type of call to retrieve the handler for",
            },
            "objection_key": {
                "type": "string",
                "description": (
                    "Objection identifier (e.g., 'too_expensive', 'not_interested', "
                    "'need_time')"
                ),
            },
        },
        "required": ["call_type", "objection_key"],
        "handler": get_objection_handler,
    },
    {
        "name": "list_objections",
        "description": (
            "List every objection handler in the script so the agent knows which "
            "concerns are covered."
        ),
        "parameters": {
            "call_type": {
                "type": "string",
                "description": "Type of call to get objections for",
            },
        },
        "required": ["call_type"],
        "handler": list_objections,
    },
]

"""
CRM Call History Tools

Provides tools for managing call history and logging call summaries in Supabase.
These tools allow the voice agent to access previous conversations and log new calls.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from voice_agent.utils.db import get_supabase_client


logger = logging.getLogger(__name__)


# Tool implementations

def get_call_history(customer_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    Get recent call history for a customer.
    
    Args:
        customer_id: Customer identifier
        limit: Number of recent calls to retrieve (default 5)
        
    Returns:
        List of recent calls with summaries
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("calls").select(
            "id, call_date, duration_seconds, call_type, outcome, summary, sentiment, next_action, next_action_date"
        ).eq("customer_id", customer_id).order("call_date", desc=True).limit(limit).execute()
        
        calls = response.data
        
        return {
            "success": True,
            "customer_id": customer_id,
            "call_count": len(calls),
            "calls": calls,
        }
        
    except Exception as e:
        logger.error(f"Error fetching call history: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def get_last_call(customer_id: str) -> Dict[str, Any]:
    """
    Get the most recent call for a customer.
    
    Args:
        customer_id: Customer identifier
        
    Returns:
        Most recent call details or message if no calls found
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("calls").select(
            "id, call_date, duration_seconds, call_type, outcome, summary, sentiment, next_action, next_action_date"
        ).eq("customer_id", customer_id).order("call_date", desc=True).limit(1).execute()
        
        if response.data:
            return {
                "found": True,
                **response.data[0]
            }
        else:
            return {
                "found": False,
                "message": "No previous calls found for this customer",
            }
        
    except Exception as e:
        logger.error(f"Error fetching last call: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def save_call_summary(
    customer_id: str,
    summary: str,
    outcome: Optional[str] = None,
    call_type: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    sentiment: Optional[str] = None,
    next_action: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save a summary of the current call.
    
    Args:
        customer_id: Customer identifier
        summary: Brief summary of the call (required)
        outcome: Call outcome (interested, not_interested, callback, demo_scheduled, etc.)
        call_type: Type of call (inbound_support, outbound_cold_call, outbound_followup)
        duration_seconds: Call duration in seconds
        sentiment: Customer sentiment (positive, neutral, negative)
        next_action: What should happen next
        
    Returns:
        Confirmation with call_id
    """
    try:
        supabase = get_supabase_client()
        
        # Verify customer exists
        cust_check = supabase.table("customers").select("id, org_id").eq("id", customer_id).execute()
        if not cust_check.data:
            return {
                "error": f"Customer not found: {customer_id}",
            }
        
        org_id = cust_check.data[0]["org_id"]
        
        # Generate call ID
        call_id = str(uuid.uuid4())
        
        call_data = {
            "id": call_id, # Schema uses 'id' but tool returns 'call_id'. We insert into 'id'.
            "org_id": org_id,
            "customer_id": customer_id,
            "call_date": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "call_type": call_type,
            "outcome": outcome,
            "summary": summary,
            "sentiment": sentiment,
            "next_action": next_action,
        }
        
        # Remove None values
        call_data = {k: v for k, v in call_data.items() if v is not None}
        
        supabase.table("calls").insert(call_data).execute()
        
        return {
            "success": True,
            "call_id": call_id,
            "customer_id": customer_id,
            "message": "Call summary saved successfully",
        }
        
    except Exception as e:
        logger.error(f"Error saving call summary: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def update_call_transcript(call_id: str, transcript: str) -> Dict[str, Any]:
    """
    Update the full transcript for a call.
    
    Args:
        call_id: Call identifier
        transcript: Full conversation transcript
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        # Verify call exists
        call_check = supabase.table("calls").select("id").eq("id", call_id).execute()
        if not call_check.data:
            return {
                "error": f"Call not found: {call_id}",
            }
        
        response = supabase.table("calls").update({"transcript": transcript}).eq("id", call_id).execute()
        
        if response.data:
            return {
                "success": True,
                "call_id": call_id,
                "message": "Transcript updated successfully",
            }
        else:
             return {"error": "Failed to update transcript"}
        
    except Exception as e:
        logger.error(f"Error updating transcript: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def get_calls_by_outcome(outcome: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get recent calls with a specific outcome.
    
    Args:
        outcome: Call outcome to filter by
        limit: Maximum number of results (default 10)
        
    Returns:
        List of calls with matching outcome
    """
    try:
        supabase = get_supabase_client()
        
        # Join with customers table to get name/company
        # Supabase-py supports joins via select syntax: select("*, customers(*)")
        response = supabase.table("calls").select(
            "call_id:id, customer_id, call_date, summary, customers(name, company, phone)"
        ).eq("outcome", outcome).order("call_date", desc=True).limit(limit).execute()
        
        calls = []
        for row in response.data:
            customer = row.get("customers") or {}
            calls.append({
                "call_id": row["call_id"],
                "customer_id": row["customer_id"],
                "call_date": row["call_date"],
                "summary": row["summary"],
                "customer_name": customer.get("name"),
                "customer_company": customer.get("company"),
                "customer_phone": customer.get("phone"),
            })
        
        return {
            "success": True,
            "outcome": outcome,
            "count": len(calls),
            "calls": calls,
        }
        
    except Exception as e:
        logger.error(f"Error fetching calls by outcome: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def create_call_record(
    call_id: str,
    customer_id: str,
    call_type: str = "inbound_support",
    outcome: Optional[str] = None,
    summary: Optional[str] = None,
    customer_phone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new call record with a specific ID.
    
    Args:
        call_id: Unique call identifier (e.g. Twilio CallSid)
        customer_id: Customer identifier
        call_type: Type of call
        outcome: Initial outcome
        summary: Initial summary
        customer_phone: Customer phone number (optional, used if customer_id is unknown)
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        # Check if call already exists
        existing = supabase.table("calls").select("id").eq("id", call_id).execute()
        if existing.data:
            return {
                "success": True,
                "call_id": call_id,
                "message": "Call record already exists",
            }

        # Resolve customer_id if unknown
        final_customer_id = customer_id
        if (not final_customer_id or final_customer_id == "unknown") and customer_phone:
            # Try to find customer by phone
            # Clean phone
            clean_phone = customer_phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
            cust_search = supabase.table("customers").select("id").or_(
                f"phone.eq.{customer_phone},phone.eq.{clean_phone}"
            ).execute()
            
            if cust_search.data:
                final_customer_id = cust_search.data[0]["id"]
            else:
                # Create new customer
                # Need org_id
                org_response = supabase.table("organizations").select("id").limit(1).execute()
                if org_response.data:
                    org_id = org_response.data[0]["id"]
                    new_cust_data = {
                        "org_id": org_id,
                        "phone": customer_phone,
                        "name": "Unknown Caller",
                        "status": "new",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    }
                    cust_create = supabase.table("customers").insert(new_cust_data).execute()
                    if cust_create.data:
                        final_customer_id = cust_create.data[0]["id"]
        
        if not final_customer_id or final_customer_id == "unknown":
             return {"error": "Valid customer_id required and could not be resolved from phone."}

        # Need org_id from customer
        cust_check = supabase.table("customers").select("org_id").eq("id", final_customer_id).execute()
        if not cust_check.data:
             return {"error": f"Customer {final_customer_id} not found, cannot create call record."}
        
        org_id = cust_check.data[0]["org_id"]

        call_data = {
            "id": call_id,
            "org_id": org_id,
            "customer_id": final_customer_id,
            "call_date": datetime.now().isoformat(),
            "call_type": call_type,
            "outcome": outcome,
            "summary": summary,
        }
        
        # Remove None
        call_data = {k: v for k, v in call_data.items() if v is not None}
        
        supabase.table("calls").insert(call_data).execute()
        
        return {
            "success": True,
            "call_id": call_id,
            "message": "Call record created successfully",
        }
        
    except Exception as e:
        logger.error(f"Error creating call record: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def get_transcript(call_id: str) -> Dict[str, Any]:
    """
    Retrieve the transcript for a specific call.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Transcript text or error
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("calls").select("transcript").eq("id", call_id).execute()
        
        if response.data:
            return {
                "success": True,
                "call_id": call_id,
                "transcript": response.data[0].get("transcript") or "",
            }
        else:
            return {
                "error": f"Call not found: {call_id}",
            }
            
    except Exception as e:
        logger.error(f"Error fetching transcript: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


# Tool definitions for MCP server

TOOL_DEFINITIONS = [
    {
        "name": "get_call_history",
        "description": (
            "Get recent call history for a customer. Use at the start of a call to "
            "understand prior interactions."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer identifier (UUID)"
            },
            "limit": {
                "type": "integer",
                "description": "Number of recent calls to retrieve (default 5)"
            }
        },
        "required": ["customer_id"],
        "handler": get_call_history,
    },
    {
        "name": "get_last_call",
        "description": (
            "Get the most recent call for a customer to quickly reference the "
            "latest conversation."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer identifier (UUID)"
            }
        },
        "required": ["customer_id"],
        "handler": get_last_call,
    },
    {
        "name": "save_call_summary",
        "description": (
            "Save a summary of the current call so follow-up actions are logged "
            "consistently."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer identifier (UUID)"
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of what happened in the call"
            },
            "outcome": {
                "type": "string",
                "description": (
                    "Call outcome (interested, not_interested, callback, demo_scheduled, "
                    "etc.)"
                )
            },
            "call_type": {
                "type": "string",
                "description": "Type: inbound_support, outbound_cold_call, outbound_followup"
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Call duration in seconds"
            },
            "sentiment": {
                "type": "string",
                "description": "Customer sentiment: positive, neutral, negative"
            },
            "next_action": {
                "type": "string",
                "description": "What should happen next"
            }
        },
        "required": ["customer_id", "summary"],
        "handler": save_call_summary,
    },
    {
        "name": "update_call_transcript",
        "description": "Store the full conversation transcript for a call.",
        "parameters": {
            "call_id": {
                "type": "string",
                "description": "Call identifier (UUID)"
            },
            "transcript": {
                "type": "string",
                "description": "Full conversation text"
            }
        },
        "required": ["call_id", "transcript"],
        "handler": update_call_transcript,
    },
    {
        "name": "get_calls_by_outcome",
        "description": (
            "Find recent calls with a specific outcome, such as interested customers "
            "or demo requests."
        ),
        "parameters": {
            "outcome": {
                "type": "string",
                "description": "Outcome to filter by"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 10)"
            }
        },
        "required": ["outcome"],
        "handler": get_calls_by_outcome,
    },
]

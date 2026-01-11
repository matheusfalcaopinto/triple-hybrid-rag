"""
CRM Customer Facts Tools

Provides tools for storing and retrieving learned information about customers in Supabase.
These tools enable the agent to build long-term memory about customer preferences,
pain points, business context, and personal details.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from voice_agent.utils.db import get_supabase_client


logger = logging.getLogger(__name__)


# Tool implementations

def get_customer_facts(
    customer_id: str = None,
    phone: str = None,  # Alias: auto-lookup customer_id by phone
) -> Dict[str, Any]:
    """
    Get all learned facts about a customer.
    
    Args:
        customer_id: Customer identifier
        phone: Alternative - phone number (will lookup customer_id automatically)
        
    Returns:
        List of facts organized by type
    """
    # Handle phone as alias - lookup customer_id
    if customer_id is None and phone is not None:
        try:
            from .crm_customer import get_customer_by_phone
            result = get_customer_by_phone(phone)
            if result.get("found") and result.get("id"):
                customer_id = result["id"]
            else:
                return {"error": f"Customer not found with phone: {phone}"}
        except Exception as e:
            return {"error": f"Failed to lookup customer by phone: {e}"}
    
    if not customer_id:
        return {"error": "Missing required parameter: customer_id (or phone)"}
    
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("customer_facts").select(
            "id, fact_type, content, confidence, learned_from_call, created_at"
        ).eq("customer_id", customer_id).order("created_at", desc=True).execute()
        
        rows = response.data
        
        # Organize facts by type
        facts_by_type: dict[str, list[Dict[str, Any]]] = {
            "preference": [],
            "pain_point": [],
            "business_context": [],
            "personal": [],
            "objection": [],
            "other": [],
        }
        
        all_facts = []
        for row in rows:
            fact = {
                "fact_id": row["id"], # Schema uses 'id'
                "fact_type": row["fact_type"],
                "content": row["content"],
                "confidence": row["confidence"],
                "learned_from_call": row["learned_from_call"],
                "created_at": row["created_at"],
            }
            all_facts.append(fact)
            
            # Add to type-specific list
            fact_type = row["fact_type"] or "other"
            if fact_type in facts_by_type:
                facts_by_type[fact_type].append(fact)
            else:
                facts_by_type["other"].append(fact)
        
        return {
            "success": True,
            "customer_id": customer_id,
            "total_facts": len(all_facts),
            "facts": all_facts,
            "facts_by_type": facts_by_type,
        }
        
    except Exception as e:
        logger.error(f"Error fetching customer facts: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def add_customer_fact(
    customer_id: str = None,
    fact_type: str = None,
    content: str = None,
    confidence: float = 1.0,
    learned_from_call: Optional[str] = None,
    phone: str = None,  # Alias: auto-lookup customer_id by phone
) -> Dict[str, Any]:
    """
    Store a new fact about a customer.
    
    Args:
        customer_id: Customer identifier
        fact_type: Type of fact (preference, pain_point, business_context, personal, objection)
        content: The actual fact/information
        confidence: Confidence level 0.0-1.0 (default 1.0)
        learned_from_call: Call ID where this was learned (optional)
        phone: Alternative - phone number (will lookup customer_id automatically)
        
    Returns:
        Confirmation with fact_id
    """
    # Handle phone as alias - lookup customer_id
    if customer_id is None and phone is not None:
        try:
            from .crm_customer import get_customer_by_phone
            result = get_customer_by_phone(phone)
            if result.get("found") and result.get("id"):
                customer_id = result["id"]
            else:
                return {"error": f"Customer not found with phone: {phone}"}
        except Exception as e:
            return {"error": f"Failed to lookup customer by phone: {e}"}
    
    if not customer_id:
        return {"error": "Missing required parameter: customer_id (or phone)"}
    if not fact_type:
        return {"error": "Missing required parameter: fact_type"}
    if not content:
        return {"error": "Missing required parameter: content"}
    
    # Valid fact types
    valid_types = ["preference", "pain_point", "business_context", "personal", "objection"]
    
    if fact_type not in valid_types:
        return {
            "error": f"Invalid fact_type: {fact_type}",
            "valid_types": valid_types,
        }
    
    if not (0.0 <= confidence <= 1.0):
        return {
            "error": "Confidence must be between 0.0 and 1.0",
        }
    
    try:
        supabase = get_supabase_client()
        
        # Verify customer exists and get org_id
        cust_check = supabase.table("customers").select("id, org_id").eq("id", customer_id).execute()
        if not cust_check.data:
            return {
                "error": f"Customer not found: {customer_id}",
            }
        
        org_id = cust_check.data[0]["org_id"]
        
        # Generate fact ID
        fact_id = str(uuid.uuid4())
        
        fact_data = {
            "id": fact_id,
            "org_id": org_id,
            "customer_id": customer_id,
            "fact_type": fact_type,
            "content": content,
            "confidence": confidence,
            "learned_from_call": learned_from_call,
            "created_at": datetime.now().isoformat(),
        }
        
        # Remove None
        fact_data = {k: v for k, v in fact_data.items() if v is not None}
        
        supabase.table("customer_facts").insert(fact_data).execute()
        
        return {
            "success": True,
            "fact_id": fact_id,
            "customer_id": customer_id,
            "fact_type": fact_type,
            "content": content,
            "message": "Fact stored successfully",
        }
        
    except Exception as e:
        logger.error(f"Error adding customer fact: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def get_facts_by_type(
    customer_id: str,
    fact_type: str
) -> Dict[str, Any]:
    """
    Get all facts of a specific type for a customer.
    
    Args:
        customer_id: Customer identifier
        fact_type: Type to filter by (preference, pain_point, business_context, etc.)
        
    Returns:
        List of facts matching the type
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("customer_facts").select(
            "id, content, confidence, learned_from_call, created_at"
        ).eq("customer_id", customer_id).eq("fact_type", fact_type).order("created_at", desc=True).execute()
        
        facts = []
        for row in response.data:
            facts.append({
                "fact_id": row["id"],
                "content": row["content"],
                "confidence": row["confidence"],
                "learned_from_call": row["learned_from_call"],
                "created_at": row["created_at"],
            })
        
        return {
            "success": True,
            "customer_id": customer_id,
            "fact_type": fact_type,
            "count": len(facts),
            "facts": facts,
        }
        
    except Exception as e:
        logger.error(f"Error fetching facts by type: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def search_customers_by_fact(
    query: str,
    fact_type: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Find customers who have facts matching a search term.
    
    Args:
        query: Search term to match in fact content
        fact_type: Optional type filter (preference, pain_point, etc.)
        limit: Maximum results (default 10)
        
    Returns:
        List of customers with matching facts
    """
    try:
        supabase = get_supabase_client()
        
        # Build query
        db_query = supabase.table("customer_facts").select(
            "fact_type, content, customers(id, name, company, phone, updated_at)"
        ).ilike("content", f"%{query}%")
        
        if fact_type:
            db_query = db_query.eq("fact_type", fact_type)
            
        # Order by customer updated_at is tricky via join sort in simple client.
        # We'll sort in python or just sort by fact creation.
        # Let's limit first.
        response = db_query.limit(limit).execute()
        
        customers = []
        for row in response.data:
            customer = row.get("customers") or {}
            if not customer: continue # Should not happen due to FK
            
            customers.append({
                "customer_id": customer.get("id"),
                "name": customer.get("name"),
                "company": customer.get("company"),
                "phone": customer.get("phone"),
                "matching_fact_type": row["fact_type"],
                "matching_fact_content": row["content"],
            })
        
        return {
            "success": True,
            "query": query,
            "fact_type_filter": fact_type,
            "count": len(customers),
            "customers": customers,
        }
        
    except Exception as e:
        logger.error(f"Error searching customers by fact: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def delete_customer_fact(fact_id: str) -> Dict[str, Any]:
    """
    Delete a fact that is incorrect or no longer relevant.
    
    Args:
        fact_id: Fact identifier
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        # Verify fact exists
        fact_check = supabase.table("customer_facts").select("id, content").eq("id", fact_id).execute()
        
        if not fact_check.data:
            return {
                "error": f"Fact not found: {fact_id}",
            }
        
        row = fact_check.data[0]
        
        # Delete fact
        supabase.table("customer_facts").delete().eq("id", fact_id).execute()
        
        return {
            "success": True,
            "fact_id": fact_id,
            "deleted_content": row["content"],
            "message": "Fact deleted successfully",
        }
        
    except Exception as e:
        logger.error(f"Error deleting customer fact: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


# Tool definitions for MCP server

TOOL_DEFINITIONS = [
    {
        "name": "get_customer_facts",
        "description": (
            "Get all learned facts about a customer to prepare before "
            "starting a conversation. Identify customer by customer_id OR phone."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer identifier (UUID) - use this OR phone"
            },
            "phone": {
                "type": "string",
                "description": "Customer phone number - use this OR customer_id (will auto-lookup)"
            }
        },
        "required": [],  # Either customer_id or phone required - validated in handler
        "handler": get_customer_facts,
    },
    {
        "name": "add_customer_fact",
        "description": (
            "Store a new fact discovered about the customer during a call or "
            "interaction. Identify customer by customer_id OR phone."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer identifier (UUID) - use this OR phone"
            },
            "phone": {
                "type": "string",
                "description": "Customer phone number - use this OR customer_id (will auto-lookup)"
            },
            "fact_type": {
                "type": "string",
                "description": (
                    "Type: preference, pain_point, business_context, personal, or "
                    "objection"
                )
            },
            "content": {
                "type": "string",
                "description": "The fact or insight to store"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level 0.0-1.0 (default 1.0)"
            },
            "learned_from_call": {
                "type": "string",
                "description": "Call ID where this was learned (optional)"
            }
        },
        "required": ["fact_type", "content"],  # customer_id OR phone required - validated in handler
        "handler": add_customer_fact,
    },
    {
        "name": "get_facts_by_type",
        "description": (
            "List all facts of a given type for a customer, such as every "
            "recorded preference."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer identifier (UUID)"
            },
            "fact_type": {
                "type": "string",
                "description": (
                    "Type to filter: preference, pain_point, business_context, "
                    "personal, objection"
                )
            }
        },
        "required": ["customer_id", "fact_type"],
        "handler": get_facts_by_type,
    },
    {
        "name": "search_customers_by_fact",
        "description": (
            "Find customers whose stored facts contain a specific concept, such "
            "as 'interested in API access'."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search term to match in fact content"
            },
            "fact_type": {
                "type": "string",
                "description": "Optional type filter: preference, pain_point, etc."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 10)"
            }
        },
        "required": ["query"],
        "handler": search_customers_by_fact,
    },
    {
        "name": "delete_customer_fact",
        "description": (
            "Delete a fact that is incorrect, outdated, or no longer relevant."
        ),
        "parameters": {
            "fact_id": {
                "type": "string",
                "description": "Fact identifier (UUID)"
            }
        },
        "required": ["fact_id"],
        "handler": delete_customer_fact,
    },
]

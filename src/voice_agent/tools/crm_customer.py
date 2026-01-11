"""
CRM Customer Tools

Provides tools for managing customer profiles in the CRM database (Supabase).
These tools allow the voice agent to lookup, create, and update customer information.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union

from voice_agent.utils.db import get_supabase_client


logger = logging.getLogger(__name__)


# Tool implementations

def get_customer_by_phone(phone: str) -> Dict[str, Any]:
    """
    Lookup customer profile by phone number.
    
    Args:
        phone: Phone number (can be in any format)
        
    Returns:
        Customer profile dict or error message
    """
    try:
        supabase = get_supabase_client()
        
        # Clean phone number (remove spaces, dashes, parentheses)
        clean_phone = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        
        # Try exact match first
        # We use the Service Role key, so we see all customers.
        # Ideally we should filter by org_id, but for now we return the first match.
        response = supabase.table("customers").select("*").or_(
            f"phone.eq.{phone},phone.eq.{clean_phone},phone.eq.{clean_phone.replace('+', '')}"
        ).execute()
        
        if response.data:
            row = response.data[0]
            return {
                "found": True,
                **row
            }
        else:
            return {
                "found": False,
                "message": f"No customer found with phone number: {phone}",
                "phone": phone,
            }
            
    except Exception as e:
        logger.error(f"Error fetching customer by phone: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "phone": phone,
        }


def get_customer_by_id(customer_id: str) -> Dict[str, Any]:
    """
    Get full customer profile by customer ID.
    
    Args:
        customer_id: Unique customer identifier
        
    Returns:
        Customer profile dict or error message
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("customers").select("*").eq("id", customer_id).execute()
        
        if response.data:
            return {
                "found": True,
                **response.data[0]
            }
        else:
            return {
                "found": False,
                "message": f"No customer found with ID: {customer_id}",
            }
            
    except Exception as e:
        logger.error(f"Error fetching customer by ID: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def create_customer(
    phone: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    company: Optional[str] = None,
    role: Optional[str] = None,
    lead_source: Optional[str] = None,
    has_whatsapp: Optional[bool] = None,
    whatsapp_number: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new customer record.
    
    Args:
        phone: Phone number (required)
        name: Customer name (optional)
        email: Email address (optional)
        company: Company name (optional)
        role: Role/title (optional)
        lead_source: How they found us (optional)
        has_whatsapp: Whether phone has WhatsApp (optional: True/False/None)
        whatsapp_number: Alternative WhatsApp number if different from phone (optional)
        
    Returns:
        New customer profile dict with customer_id
    """
    try:
        supabase = get_supabase_client()
        
        # Check if customer already exists
        existing = supabase.table("customers").select("id").eq("phone", phone).execute()
        
        if existing.data:
            return {
                "error": f"Customer with phone {phone} already exists",
                "customer_id": existing.data[0]["id"],
                "message": "Use update_customer to modify existing customer",
            }
        
        # We need an org_id. For now, fetch the 'legacy' org or the first one.
        # In a real multi-tenant setup, this should come from context.
        org_response = supabase.table("organizations").select("id").limit(1).execute()
        if not org_response.data:
             return {"error": "No organization found to assign customer to."}
        org_id = org_response.data[0]["id"]

        customer_data = {
            "org_id": org_id,
            "phone": phone,
            "name": name,
            "email": email,
            "company": company,
            "role": role,
            "lead_source": lead_source,
            "status": "new",
            "has_whatsapp": has_whatsapp,
            "whatsapp_number": whatsapp_number,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Remove None values to let DB defaults handle them if needed (though we set most)
        customer_data = {k: v for k, v in customer_data.items() if v is not None}

        response = supabase.table("customers").insert(customer_data).execute()
        
        if response.data:
            new_customer = response.data[0]
            return {
                "success": True,
                "customer_id": new_customer["id"],
                "phone": new_customer["phone"],
                "name": new_customer["name"],
                "status": new_customer["status"],
                "message": "Customer created successfully",
            }
        else:
             return {"error": "Failed to create customer (no data returned)"}
        
    except Exception as e:
        logger.error(f"Error creating customer: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def update_customer_status(customer_id: str, status: str) -> Dict[str, Any]:
    """
    Update customer's status in the sales pipeline.
    
    Args:
        customer_id: Customer identifier
        status: New status (new, contacted, qualified, customer, lost)
        
    Returns:
        Success confirmation or error message
    """
    # Valid statuses
    valid_statuses = ["new", "contacted", "qualified", "customer", "lost"]
    
    if status not in valid_statuses:
        return {
            "error": f"Invalid status: {status}",
            "valid_statuses": valid_statuses,
        }
    
    try:
        supabase = get_supabase_client()
        
        data = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        response = supabase.table("customers").update(data).eq("id", customer_id).execute()
        
        if response.data:
            return {
                "success": True,
                "customer_id": customer_id,
                "new_status": status,
                "message": f"Status updated to '{status}'",
            }
        else:
            return {
                "error": f"Customer not found: {customer_id}",
            }
        
    except Exception as e:
        logger.error(f"Error updating customer status: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def update_customer_info(
    customer_id: str = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
    company: Optional[str] = None,
    role: Optional[str] = None,
    notes: Optional[str] = None,
    has_whatsapp: Optional[bool] = None,
    whatsapp_number: Optional[str] = None,
    phone: Optional[str] = None,  # Alias: auto-lookup customer_id by phone
) -> Dict[str, Any]:
    """
    Update customer information fields.
    
    Args:
        customer_id: Customer identifier
        name: Customer name (optional)
        email: Email address (optional)
        company: Company name (optional)
        role: Role/title (optional)
        notes: Additional notes (optional)
        has_whatsapp: Whether phone has WhatsApp (optional: True/False/None)
        whatsapp_number: Alternative WhatsApp number if different from phone (optional)
        phone: Alternative - phone number (will lookup customer_id automatically)
        
    Returns:
        Success confirmation with updated fields
    """
    # Handle phone as alias - lookup customer_id
    if customer_id is None and phone is not None:
        result = get_customer_by_phone(phone)
        if result.get("found") and result.get("id"):
            customer_id = result["id"]
        else:
            return {"error": f"Customer not found with phone: {phone}"}
    
    if not customer_id:
        return {"error": "Missing required parameter: customer_id (or phone)"}
    
    try:
        supabase = get_supabase_client()
        
        updates = {}
        if name is not None: updates["name"] = name
        if email is not None: updates["email"] = email
        if company is not None: updates["company"] = company
        if role is not None: updates["role"] = role
        if notes is not None: updates["notes"] = notes
        if has_whatsapp is not None: updates["has_whatsapp"] = has_whatsapp
        if whatsapp_number is not None: updates["whatsapp_number"] = whatsapp_number
        
        if not updates:
            return {
                "error": "No fields provided to update",
                "message": "Provide at least one field to update",
            }
        
        updates["updated_at"] = datetime.now().isoformat()
        
        response = supabase.table("customers").update(updates).eq("id", customer_id).execute()
        
        if response.data:
            return {
                "success": True,
                "customer_id": customer_id,
                "updated_fields": updates,
                "message": "Customer information updated successfully",
            }
        else:
            return {
                "error": f"Customer not found: {customer_id}",
            }
        
    except Exception as e:
        logger.error(f"Error updating customer info: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


def search_customers(
    query: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search customers by name, company, or email.
    
    Args:
        query: Search term (matches name, company, or email)
        limit: Maximum number of results (default 10)
        
    Returns:
        List of matching customers
    """
    try:
        supabase = get_supabase_client()
        
        # Supabase 'or' filter with ilike
        search_filter = f"name.ilike.%{query}%,company.ilike.%{query}%,email.ilike.%{query}%"
        
        response = supabase.table("customers").select(
            "id, phone, name, email, company, role, status, has_whatsapp, whatsapp_number"
        ).or_(search_filter).order("updated_at", desc=True).limit(limit).execute()
        
        customers = []
        for row in response.data:
            # Map 'id' back to 'customer_id' for compatibility if needed, 
            # but we should probably standardize on 'customer_id' in the output dict
            # or just 'id'. The tool definition says 'customer_id'.
            row["customer_id"] = row["id"]
            customers.append(row)
        
        return {
            "success": True,
            "count": len(customers),
            "query": query,
            "customers": customers,
        }
        
    except Exception as e:
        logger.error(f"Error searching customers: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


# Tool definitions for MCP server

TOOL_DEFINITIONS = [
    {
        "name": "get_customer_by_phone",
        "description": (
            "Lookup a customer profile by phone number so inbound callers can be "
            "identified quickly."
        ),
        "parameters": {
            "phone": {
                "type": "string",
                "description": "Phone number with country code (e.g., '+55 11 99999-0001')"
            }
        },
        "required": ["phone"],
        "handler": get_customer_by_phone,
    },
    {
        "name": "get_customer_by_id",
        "description": (
            "Retrieve a full customer profile when the customer_id is already "
            "known."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Unique customer identifier (UUID)"
            }
        },
        "required": ["customer_id"],
        "handler": get_customer_by_id,
    },
    {
        "name": "create_customer",
        "description": (
            "Create a new customer record for a first-time caller using the "
            "provided phone number."
        ),
        "parameters": {
            "phone": {
                "type": "string",
                "description": "Phone number with country code (required)"
            },
            "name": {
                "type": "string",
                "description": "Customer's full name (optional)"
            },
            "email": {
                "type": "string",
                "description": "Email address (optional)"
            },
            "company": {
                "type": "string",
                "description": "Company name (optional)"
            },
            "role": {
                "type": "string",
                "description": "Job title or role (optional)"
            },
            "lead_source": {
                "type": "string",
                "description": (
                    "How they found us: website_form, referral, cold_call, etc. "
                    "(optional)"
                )
            },
            "has_whatsapp": {
                "type": "boolean",
                "description": "Whether the phone number has WhatsApp (optional)"
            },
            "whatsapp_number": {
                "type": "string",
                "description": "Alternate WhatsApp number if different from phone"
            }
        },
        "required": ["phone"],
        "handler": create_customer,
    },
    {
        "name": "update_customer_status",
        "description": (
            "Update the customer's sales status (new, contacted, qualified, "
            "customer, or lost)."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer identifier (UUID)"
            },
            "status": {
                "type": "string",
                "description": "New status: new, contacted, qualified, customer, or lost"
            }
        },
        "required": ["customer_id", "status"],
        "handler": update_customer_status,
    },
    {
        "name": "update_customer_info",
        "description": (
            "Update customer details such as name, contact info, notes, or "
            "WhatsApp metadata as new insights appear. You can identify the customer "
            "by customer_id OR phone number."
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
            "name": {
                "type": "string",
                "description": "Customer's full name (optional)"
            },
            "email": {
                "type": "string",
                "description": "Email address (optional)"
            },
            "company": {
                "type": "string",
                "description": "Company name (optional)"
            },
            "role": {
                "type": "string",
                "description": "Job title or role (optional)"
            },
            "notes": {
                "type": "string",
                "description": "Additional notes about the customer (optional)"
            },
            "has_whatsapp": {
                "type": "boolean",
                "description": "Whether the phone number has WhatsApp (optional)"
            },
            "whatsapp_number": {
                "type": "string",
                "description": "Alternate WhatsApp number if different from phone"
            }
        },
        "required": [],  # Either customer_id or phone required - validated in handler
        "handler": update_customer_info,
    },
    {
        "name": "search_customers",
        "description": (
            "Search customers by name, company, or email and return the top "
            "matches."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search term to match against name, company, or email"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 10)"
            }
        },
        "required": ["query"],
        "handler": search_customers,
    },
]

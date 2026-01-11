"""
CRM Knowledge Base Tools

Provides tools for searching and managing company knowledge base in Supabase.
Uses Supabase text search or vector search (if enabled) for retrieval.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from voice_agent.utils.db import get_supabase_client


logger = logging.getLogger(__name__)


# Tool implementations

def search_knowledge_base(
    query: str,
    category: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Search the knowledge base.
    
    Args:
        query: Search query (what the customer is asking about)
        category: Filter by category (pricing, technical, faq, product, etc.) (optional)
        limit: Maximum number of results to return (default 5)
        
    Returns:
        List of relevant knowledge base entries with ranking
    """
    try:
        supabase = get_supabase_client()
        
        # Use textSearch on content. 
        # Note: This uses 'english' config by default in Supabase client usually.
        # If we need multi-language, we might need to be more specific or use ilike.
        # For now, we try textSearch on 'content' OR 'title'.
        # Supabase client 'or' with textSearch is tricky.
        # We will use a simple ilike for broad matching if textSearch is too strict,
        # but textSearch is better for "keywords".
        # Let's use 'ilike' for now to ensure we match partial words easily without FTS config issues.
        # "title.ilike.%query%,content.ilike.%query%"
        
        search_filter = f"title.ilike.%{query}%,content.ilike.%{query}%,keywords.ilike.%{query}%"
        
        db_query = supabase.table("knowledge_base").select(
            "id, category, title, content, keywords, source_document, created_at, updated_at, access_count"
        ).or_(search_filter)
        
        if category:
            db_query = db_query.eq("category", category)
            
        # Order by access_count desc as a proxy for relevance if we don't have FTS rank
        response = db_query.order("access_count", desc=True).limit(limit).execute()
        
        results = []
        for row in response.data:
            # Increment access count (async or fire-and-forget ideally, but here sync)
            # We do it in a separate call to not block the read? 
            # Actually, we can just fire it and ignore result, or skip it for speed.
            # Let's skip incrementing for now to save latency, or do it.
            # supabase.table("knowledge_base").update({"access_count": row["access_count"] + 1}).eq("id", row["id"]).execute()
            
            results.append({
                "kb_id": row["id"],
                "category": row["category"],
                "title": row["title"],
                "content": row["content"],
                "keywords": row["keywords"],
                "source_document": row["source_document"],
                "access_count": row["access_count"],
                "relevance_rank": 0, # No rank with ilike
            })
        
        return {
            "success": True,
            "query": query,
            "category": category,
            "result_count": len(results),
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "query": query,
            "category": category,
        }


def get_knowledge_by_category(category: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get all knowledge base entries in a specific category.
    
    Args:
        category: Category to retrieve (pricing, technical, faq, product, etc.)
        limit: Maximum number of entries to return (default 10)
        
    Returns:
        List of knowledge base entries in the category
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("knowledge_base").select(
            "id, category, title, content, keywords, source_document, created_at, updated_at, access_count"
        ).eq("category", category).order("access_count", desc=True).limit(limit).execute()
        
        results = []
        for row in response.data:
            results.append({
                "kb_id": row["id"],
                "category": row["category"],
                "title": row["title"],
                "content": row["content"],
                "keywords": row["keywords"],
                "source_document": row["source_document"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "access_count": row["access_count"],
            })
        
        return {
            "success": True,
            "category": category,
            "result_count": len(results),
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Error fetching knowledge by category: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "category": category,
        }


def get_knowledge_by_id(kb_id: str) -> Dict[str, Any]:
    """
    Get a specific knowledge base entry by ID.
    
    Args:
        kb_id: Knowledge base entry UUID
        
    Returns:
        Knowledge base entry details
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("knowledge_base").select("*").eq("id", kb_id).execute()
        
        if response.data:
            row = response.data[0]
            # Increment access count
            # supabase.table("knowledge_base").update({"access_count": row["access_count"] + 1}).eq("id", kb_id).execute()
            
            return {
                "found": True,
                "kb_id": row["id"],
                "category": row["category"],
                "title": row["title"],
                "content": row["content"],
                "keywords": row["keywords"],
                "source_document": row["source_document"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "access_count": row["access_count"],
            }
        else:
            return {
                "found": False,
                "message": f"No knowledge base entry found with kb_id: {kb_id}",
                "kb_id": kb_id,
            }
        
    except Exception as e:
        logger.error(f"Error fetching knowledge by ID: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "kb_id": kb_id,
        }


def add_knowledge_item(
    category: str,
    title: str,
    content: str,
    keywords: Optional[str] = None,
    source_document: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add a new entry to the knowledge base.
    
    Args:
        category: Category (pricing, technical, faq, product, etc.)
        title: Title of the knowledge base entry
        content: Full content/answer
        keywords: Space-separated keywords for search (optional)
        source_document: Source document name if imported (optional)
        
    Returns:
        Created knowledge base entry details
    """
    try:
        supabase = get_supabase_client()
        
        # Need org_id. Fetch default.
        org_response = supabase.table("organizations").select("id").limit(1).execute()
        if not org_response.data:
             return {"error": "No organization found."}
        org_id = org_response.data[0]["id"]
        
        kb_id = str(uuid.uuid4())
        
        data = {
            "id": kb_id,
            "org_id": org_id,
            "category": category,
            "title": title,
            "content": content,
            "keywords": keywords,
            "source_document": source_document,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Remove None
        data = {k: v for k, v in data.items() if v is not None}
        
        supabase.table("knowledge_base").insert(data).execute()
        
        return {
            "success": True,
            "kb_id": kb_id,
            "category": category,
            "title": title,
            "message": "Knowledge base entry created successfully",
        }
        
    except Exception as e:
        logger.error(f"Error adding knowledge item: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "category": category,
            "title": title,
        }


def update_knowledge_item(
    kb_id: str,
    title: Optional[str] = None,
    content: Optional[str] = None,
    keywords: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing knowledge base entry.
    
    Args:
        kb_id: Knowledge base entry UUID
        title: New title (optional)
        content: New content (optional)
        keywords: New keywords (optional)
        category: New category (optional)
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        updates = {}
        if title is not None: updates["title"] = title
        if content is not None: updates["content"] = content
        if keywords is not None: updates["keywords"] = keywords
        if category is not None: updates["category"] = category
        
        if not updates:
            return {
                "success": False,
                "message": "No fields to update",
                "kb_id": kb_id,
            }
        
        updates["updated_at"] = datetime.now().isoformat()
        
        response = supabase.table("knowledge_base").update(updates).eq("id", kb_id).execute()
        
        if response.data:
            return {
                "success": True,
                "kb_id": kb_id,
                "message": "Knowledge base entry updated successfully",
            }
        else:
            return {
                "success": False,
                "message": f"No knowledge base entry found with kb_id: {kb_id}",
                "kb_id": kb_id,
            }
        
    except Exception as e:
        logger.error(f"Error updating knowledge item: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "kb_id": kb_id,
        }


def delete_knowledge_item(kb_id: str) -> Dict[str, Any]:
    """
    Delete a knowledge base entry.
    
    Args:
        kb_id: Knowledge base entry UUID to delete
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        # Verify exists
        check = supabase.table("knowledge_base").select("id").eq("id", kb_id).execute()
        if not check.data:
            return {
                "success": False,
                "message": f"No knowledge base entry found with kb_id: {kb_id}",
                "kb_id": kb_id,
            }
        
        supabase.table("knowledge_base").delete().eq("id", kb_id).execute()
        
        return {
            "success": True,
            "kb_id": kb_id,
            "message": "Knowledge base entry deleted",
        }
        
    except Exception as e:
        logger.error(f"Error deleting knowledge item: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "kb_id": kb_id,
        }


def list_categories() -> Dict[str, Any]:
    """
    List all available knowledge base categories with entry counts.
    
    Returns:
        List of categories with entry counts
    """
    try:
        supabase = get_supabase_client()
        
        # Group by category. Supabase client doesn't support GROUP BY easily.
        # We fetch all categories and count in python, or use an RPC if available.
        # For small KB, python is fine.
        response = supabase.table("knowledge_base").select("category").execute()
        
        counts = {}
        for row in response.data:
            cat = row["category"]
            counts[cat] = counts.get(cat, 0) + 1
            
        categories = [{"category": cat, "entry_count": count} for cat, count in counts.items()]
        categories.sort(key=lambda x: x["entry_count"], reverse=True)
        
        return {
            "success": True,
            "category_count": len(categories),
            "categories": categories,
        }
        
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        return {
            "error": f"Database error: {str(e)}",
        }


TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the knowledge base for answers to customer questions and "
            "return the most relevant entries."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search query describing the customer's question",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
            },
        },
        "required": ["query"],
        "handler": search_knowledge_base,
    },
    {
        "name": "get_knowledge_by_category",
        "description": (
            "Retrieve all entries in a category such as pricing, FAQ, or "
            "product details."
        ),
        "parameters": {
            "category": {
                "type": "string",
                "description": "Category to retrieve (pricing, technical, faq, etc.)",
            },
        },
        "required": ["category"],
        "handler": get_knowledge_by_category,
    },
    {
        "name": "get_knowledge_by_id",
        "description": (
            "Fetch a specific knowledge base entry by its identifier when a "
            "precise answer is needed."
        ),
        "parameters": {
            "kb_id": {
                "type": "string",
                "description": "Knowledge base entry UUID",
            },
        },
        "required": ["kb_id"],
        "handler": get_knowledge_by_id,
    },
    {
        "name": "add_knowledge_item",
        "description": (
            "Add a new knowledge entry so fresh company or product information "
            "is searchable."
        ),
        "parameters": {
            "category": {
                "type": "string",
                "description": "Category (pricing, technical, faq, product, etc.)",
            },
            "content": {
                "type": "string",
                "description": "Full content or answer",
            },
            "keywords": {
                "type": "string",
                "description": "Space-separated keywords for search (optional)",
            },
            "source_document": {
                "type": "string",
                "description": "Source document name if imported (optional)",
            },
        },
        "required": ["category", "content"],
        "handler": add_knowledge_item,
    },
    {
        "name": "update_knowledge_item",
        "description": (
            "Update an existing knowledge entry and refresh the search index "
            "to reflect changes."
        ),
        "parameters": {
            "kb_id": {
                "type": "string",
                "description": "Knowledge base entry UUID",
            },
            "content": {
                "type": "string",
                "description": "New content (optional)",
            },
            "keywords": {
                "type": "string",
                "description": "New keywords (optional)",
            },
            "category": {
                "type": "string",
                "description": "New category (optional)",
            },
        },
        "required": ["kb_id"],
        "handler": update_knowledge_item,
    },
    {
        "name": "delete_knowledge_item",
        "description": (
            "Delete a knowledge base entry and remove it from the full-text "
            "search index."
        ),
        "parameters": {
            "kb_id": {
                "type": "string",
                "description": "Knowledge base entry UUID to delete",
            },
        },
        "required": ["kb_id"],
        "handler": delete_knowledge_item,
    },
    {
        "name": "list_categories",
        "description": (
            "List knowledge base categories along with entry counts to see "
            "which topics are covered."
        ),
        "parameters": {},
        "handler": list_categories,
    },
]

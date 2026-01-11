"""
CRM Action Items Tools

Provides tools for managing follow-up tasks and action items in Supabase.
These tools help the agent track what needs to be done after calls.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from voice_agent.utils.db import get_supabase_client

logger = logging.getLogger(__name__)


# Tool implementations

def get_pending_tasks(customer_id: str) -> Dict[str, Any]:
    """
    Get all pending action items for a customer.
    
    Args:
        customer_id: Customer UUID
        
    Returns:
        List of pending tasks
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("action_items").select(
            "id, customer_id, created_in_call, task_type, description, due_date, status, priority, assigned_to, completed_at"
        ).eq("customer_id", customer_id).eq("status", "pending").order("due_date").order("priority", desc=True).execute()
        
        tasks = []
        for row in response.data:
            tasks.append({
                "task_id": row["id"],
                "customer_id": row["customer_id"],
                "created_in_call": row["created_in_call"],
                "task_type": row["task_type"],
                "description": row["description"],
                "due_date": row["due_date"],
                "status": row["status"],
                "priority": row["priority"],
                "assigned_to": row["assigned_to"],
                "completed_at": row["completed_at"],
            })
        
        return {
            "success": True,
            "customer_id": customer_id,
            "task_count": len(tasks),
            "tasks": tasks,
        }
        
    except Exception as e:
        logger.error(f"Error fetching pending tasks: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "customer_id": customer_id,
        }


def get_all_pending_tasks(assigned_to: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all pending action items across all customers.
    
    Args:
        assigned_to: Filter by assignee (optional)
        
    Returns:
        List of all pending tasks
    """
    try:
        supabase = get_supabase_client()
        
        query = supabase.table("action_items").select(
            "id, customer_id, created_in_call, task_type, description, due_date, status, priority, assigned_to, completed_at, customers(name, company, phone)"
        ).eq("status", "pending")
        
        if assigned_to:
            query = query.eq("assigned_to", assigned_to)
            
        response = query.order("due_date").order("priority", desc=True).execute()
        
        tasks = []
        for row in response.data:
            customer = row.get("customers") or {}
            
            tasks.append({
                "task_id": row["id"],
                "customer_id": row["customer_id"],
                "customer_name": customer.get("name"),
                "customer_company": customer.get("company"),
                "customer_phone": customer.get("phone"),
                "created_in_call": row["created_in_call"],
                "task_type": row["task_type"],
                "description": row["description"],
                "due_date": row["due_date"],
                "status": row["status"],
                "priority": row["priority"],
                "assigned_to": row["assigned_to"],
                "completed_at": row["completed_at"],
            })
        
        return {
            "success": True,
            "assigned_to": assigned_to,
            "task_count": len(tasks),
            "tasks": tasks,
        }
        
    except Exception as e:
        logger.error(f"Error fetching all pending tasks: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "assigned_to": assigned_to,
        }


def create_task(
    customer_id: str,
    description: str,
    task_type: Optional[str] = None,
    due_date: Optional[str] = None,
    priority: str = "medium",
    assigned_to: Optional[str] = None,
    created_in_call: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new action item for a customer.
    
    Args:
        customer_id: Customer UUID
        description: What needs to be done
        task_type: Type of task (follow_up, send_email, send_proposal, demo, etc.)
        due_date: When task is due (ISO format or YYYY-MM-DD)
        priority: Task priority (low, medium, high)
        assigned_to: Who should handle this
        created_in_call: Call ID if created during a call
        
    Returns:
        Created task details
    """
    try:
        supabase = get_supabase_client()
        
        # Need org_id
        cust_check = supabase.table("customers").select("org_id").eq("id", customer_id).execute()
        if not cust_check.data:
             return {"error": f"Customer not found: {customer_id}"}
        org_id = cust_check.data[0]["org_id"]
        
        task_id = str(uuid.uuid4())
        
        data = {
            "id": task_id,
            "org_id": org_id,
            "customer_id": customer_id,
            "created_in_call": created_in_call,
            "task_type": task_type,
            "description": description,
            "due_date": due_date,
            "status": "pending",
            "priority": priority,
            "assigned_to": assigned_to,
            "created_at": datetime.now().isoformat(),
        }
        
        # Remove None
        data = {k: v for k, v in data.items() if v is not None}
        
        supabase.table("action_items").insert(data).execute()
        
        return {
            "success": True,
            "task_id": task_id,
            "customer_id": customer_id,
            "description": description,
            "task_type": task_type,
            "due_date": due_date,
            "priority": priority,
            "assigned_to": assigned_to,
            "status": "pending",
            "message": "Task created successfully",
        }
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "customer_id": customer_id,
        }


def complete_task(task_id: str) -> Dict[str, Any]:
    """
    Mark an action item as completed.
    
    Args:
        task_id: Task UUID to complete
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        # Verify exists
        check = supabase.table("action_items").select("id").eq("id", task_id).execute()
        if not check.data:
             return {
                "success": False,
                "message": f"No task found with task_id: {task_id}",
                "task_id": task_id,
            }
        
        data = {
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        }
        
        supabase.table("action_items").update(data).eq("id", task_id).execute()
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task marked as completed",
        }
        
    except Exception as e:
        logger.error(f"Error completing task: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "task_id": task_id,
        }


def update_task(
    task_id: str,
    description: Optional[str] = None,
    due_date: Optional[str] = None,
    priority: Optional[str] = None,
    assigned_to: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing action item.
    
    Args:
        task_id: Task UUID to update
        description: New description (optional)
        due_date: New due date (optional)
        priority: New priority (optional)
        assigned_to: New assignee (optional)
        status: New status (pending, in_progress, completed, cancelled) (optional)
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        updates = {}
        if description is not None: updates["description"] = description
        if due_date is not None: updates["due_date"] = due_date
        if priority is not None: updates["priority"] = priority
        if assigned_to is not None: updates["assigned_to"] = assigned_to
        if status is not None:
            updates["status"] = status
            if status == "completed":
                updates["completed_at"] = datetime.now().isoformat()
        
        if not updates:
            return {
                "success": False,
                "message": "No fields to update",
                "task_id": task_id,
            }
        
        response = supabase.table("action_items").update(updates).eq("id", task_id).execute()
        
        if response.data:
            return {
                "success": True,
                "task_id": task_id,
                "message": "Task updated successfully",
            }
        else:
            return {
                "success": False,
                "message": f"No task found with task_id: {task_id}",
                "task_id": task_id,
            }
        
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "task_id": task_id,
        }


def get_overdue_tasks(assigned_to: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all overdue action items.
    
    Args:
        assigned_to: Filter by assignee (optional)
        
    Returns:
        List of overdue tasks
    """
    try:
        supabase = get_supabase_client()
        
        now = datetime.now().isoformat()
        
        query = supabase.table("action_items").select(
            "id, customer_id, created_in_call, task_type, description, due_date, status, priority, assigned_to, completed_at, customers(name, company, phone)"
        ).eq("status", "pending").lt("due_date", now)
        
        if assigned_to:
            query = query.eq("assigned_to", assigned_to)
            
        response = query.order("due_date").execute()
        
        tasks = []
        for row in response.data:
            customer = row.get("customers") or {}
            
            tasks.append({
                "task_id": row["id"],
                "customer_id": row["customer_id"],
                "customer_name": customer.get("name"),
                "customer_company": customer.get("company"),
                "customer_phone": customer.get("phone"),
                "created_in_call": row["created_in_call"],
                "task_type": row["task_type"],
                "description": row["description"],
                "due_date": row["due_date"],
                "status": row["status"],
                "priority": row["priority"],
                "assigned_to": row["assigned_to"],
                "completed_at": row["completed_at"],
            })
        
        return {
            "success": True,
            "assigned_to": assigned_to,
            "overdue_count": len(tasks),
            "tasks": tasks,
        }
        
    except Exception as e:
        logger.error(f"Error fetching overdue tasks: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "assigned_to": assigned_to,
        }


def delete_task(task_id: str) -> Dict[str, Any]:
    """
    Delete an action item.
    
    Args:
        task_id: Task UUID to delete
        
    Returns:
        Success confirmation
    """
    try:
        supabase = get_supabase_client()
        
        # Verify exists
        check = supabase.table("action_items").select("id").eq("id", task_id).execute()
        if not check.data:
            return {
                "success": False,
                "message": f"No task found with task_id: {task_id}",
                "task_id": task_id,
            }
        
        supabase.table("action_items").delete().eq("id", task_id).execute()
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task deleted",
        }
        
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        return {
            "error": f"Database error: {str(e)}",
            "task_id": task_id,
        }


TOOL_DEFINITIONS = [
    {
        "name": "get_pending_tasks",
        "description": (
            "List pending action items for a customer to understand what needs "
            "follow-up before a call."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer UUID",
            },
        },
        "required": ["customer_id"],
        "handler": get_pending_tasks,
    },
    {
        "name": "get_all_pending_tasks",
        "description": (
            "List pending action items across customers, optionally filtered by "
            "assignee, to build daily task lists."
        ),
        "parameters": {
            "assigned_to": {
                "type": "string",
                "description": "Filter by assignee (optional)",
            },
        },
        "handler": get_all_pending_tasks,
    },
    {
        "name": "create_task",
        "description": (
            "Create a follow-up task during or after a call, such as sending a "
            "proposal or scheduling a demo."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "Customer UUID",
            },
            "description": {
                "type": "string",
                "description": "What needs to be done",
            },
            "task_type": {
                "type": "string",
                "description": "Task type (follow_up, send_email, demo, etc.)",
            },
            "due_date": {
                "type": "string",
                "description": "When the task is due (ISO format or YYYY-MM-DD)",
            },
            "priority": {
                "type": "string",
                "description": "Task priority (low, medium, high)",
                "enum": ["low", "medium", "high"],
            },
            "assigned_to": {
                "type": "string",
                "description": "Who should handle this task",
            },
            "created_in_call": {
                "type": "string",
                "description": "Call ID if created during a call",
            },
        },
        "required": ["customer_id", "description"],
        "handler": create_task,
    },
    {
        "name": "complete_task",
        "description": (
            "Mark a task as completed once the follow-up has been handled."
        ),
        "parameters": {
            "task_id": {
                "type": "string",
                "description": "Task UUID to complete",
            },
        },
        "required": ["task_id"],
        "handler": complete_task,
    },
    {
        "name": "update_task",
        "description": (
            "Update task details such as description, due date, priority, "
            "assignee, or status."
        ),
        "parameters": {
            "task_id": {
                "type": "string",
                "description": "Task UUID to update",
            },
            "description": {
                "type": "string",
                "description": "New description",
            },
            "due_date": {
                "type": "string",
                "description": "New due date (ISO format or YYYY-MM-DD)",
            },
            "priority": {
                "type": "string",
                "description": "New priority (low, medium, high)",
                "enum": ["low", "medium", "high"],
            },
            "assigned_to": {
                "type": "string",
                "description": "New assignee",
            },
            "status": {
                "type": "string",
                "description": (
                    "New status (pending, in_progress, completed, cancelled)"
                ),
                "enum": ["pending", "in_progress", "completed", "cancelled"],
            },
        },
        "required": ["task_id"],
        "handler": update_task,
    },
    {
        "name": "get_overdue_tasks",
        "description": (
            "List tasks that are past due and still pending, optionally filtered "
            "by assignee."
        ),
        "parameters": {
            "assigned_to": {
                "type": "string",
                "description": "Filter by assignee (optional)",
            },
        },
        "handler": get_overdue_tasks,
    },
    {
        "name": "delete_task",
        "description": (
            "Delete a task when it should be removed entirely (use sparingly; "
            "cancellation is often preferable)."
        ),
        "parameters": {
            "task_id": {
                "type": "string",
                "description": "Task UUID to delete",
            },
        },
        "required": ["task_id"],
        "handler": delete_task,
    },
]

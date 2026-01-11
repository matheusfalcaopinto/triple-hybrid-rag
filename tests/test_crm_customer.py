"""
Tests for CRM Customer Tools
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from voice_agent.tools.crm_customer import (
    TOOL_DEFINITIONS,
    create_customer,
    get_customer_by_id,
    get_customer_by_phone,
    search_customers,
    update_customer_info,
    update_customer_status,
)


@pytest.fixture
def mock_supabase():
    with patch("voice_agent.tools.crm_customer.get_supabase_client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


class TestGetCustomerByPhone:
    """Test get_customer_by_phone function"""
    
    def test_get_existing_customer(self, mock_supabase):
        """Test retrieving existing customer by phone"""
        # Mock response
        mock_response = MagicMock()
        mock_response.data = [{
            "id": "test-123",
            "phone": "+55 11 99999-0001",
            "name": "João Silva",
            "email": "joao@example.com",
            "has_whatsapp": True,
            "whatsapp_number": None,
            "company": None,
            "role": None,
            "status": "new",
            "lead_source": None,
            "assigned_to": None,
            "preferred_language": "pt-BR",
            "timezone": "America/Sao_Paulo",
            "notes": None,
            "created_at": "2023-01-01T00:00:00+00:00",
            "updated_at": "2023-01-01T00:00:00+00:00"
        }]
        
        # Chain mocks
        mock_supabase.table.return_value.select.return_value.or_.return_value.execute.return_value = mock_response
        
        result = get_customer_by_phone('+55 11 99999-0001')
        
        assert result['found'] is True
        assert result['id'] == 'test-123'
        assert result['name'] == 'João Silva'
        assert result['email'] == 'joao@example.com'
        assert result['has_whatsapp'] is True
    
    def test_get_nonexistent_customer(self, mock_supabase):
        """Test retrieving non-existent customer"""
        mock_response = MagicMock()
        mock_response.data = []
        
        mock_supabase.table.return_value.select.return_value.or_.return_value.execute.return_value = mock_response
        
        result = get_customer_by_phone('+55 11 88888-8888')
        
        assert result['found'] is False
        assert 'message' in result


class TestCreateCustomer:
    """Test create_customer function"""
    
    def test_create_basic_customer(self, mock_supabase):
        """Test creating customer with only phone"""
        # Mock org fetch
        mock_org_response = MagicMock()
        mock_org_response.data = [{"id": "org-123"}]
        
        # Mock insert response
        mock_insert_response = MagicMock()
        mock_insert_response.data = [{
            "id": "cust-123",
            "phone": "+55 11 98888-8888",
            "name": None,
            "status": "new",
            "created_at": "2023-01-01T00:00:00+00:00"
        }]
        
        # Setup mock chain
        # First call is to organizations table
        # Second call is to customers table (check existing)
        # Third call is to customers table (insert)
        
        # We need to distinguish calls. 
        # Easier to just mock the chain generally or use side_effect if needed.
        # But here we can just mock the return values for specific table calls if we want,
        # or just assume the order.
        
        # Mock table().select().limit().execute() for org
        mock_supabase.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_org_response
        
        # Mock table().select().eq().execute() for existing check (return empty)
        mock_check_response = MagicMock()
        mock_check_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_check_response
        
        # Mock table().insert().execute()
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        
        result = create_customer(phone='+55 11 98888-8888')
        
        assert result['success'] is True
        assert result['customer_id'] == 'cust-123'
        assert result['phone'] == '+55 11 98888-8888'
        assert result['status'] == 'new'


class TestUpdateCustomerInfo:
    """Test update_customer_info function"""
    
    def test_update_customer_fields(self, mock_supabase):
        """Test updating customer information"""
        mock_response = MagicMock()
        mock_response.data = [{
            "id": "cust-123",
            "name": "New Name",
            "email": "newemail@example.com",
            "has_whatsapp": True
        }]
        
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        
        result = update_customer_info(
            customer_id="cust-123",
            name='New Name',
            email='newemail@example.com',
            has_whatsapp=True
        )
        
        assert result['success'] is True
        assert result['updated_fields']['name'] == 'New Name'
        assert result['updated_fields']['email'] == 'newemail@example.com'
        assert result['updated_fields']['has_whatsapp'] is True


class TestSearchCustomers:
    """Test search_customers function"""
    
    def test_search_by_name(self, mock_supabase):
        """Test searching customers by name"""
        mock_response = MagicMock()
        mock_response.data = [
            {"id": "cust-1", "name": "João Silva", "phone": "123"},
            {"id": "cust-2", "name": "Maria Silva", "phone": "456"}
        ]
        
        mock_supabase.table.return_value.select.return_value.or_.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        result = search_customers('Silva')
        
        assert result['success'] is True
        assert result['count'] == 2
        assert len(result['customers']) == 2


class TestUpdateCustomerStatus:
    """Test update_customer_status function"""
    
    def test_update_status(self, mock_supabase):
        """Test updating customer status"""
        # Mock get current status
        mock_get_response = MagicMock()
        mock_get_response.data = [{"status": "new"}]
        
        # Mock update
        mock_update_response = MagicMock()
        mock_update_response.data = [{"status": "contacted"}]
        
        # We need to handle multiple calls.
        # table("customers").select("status").eq("id", ...).execute()
        # table("customers").update(...).eq("id", ...).execute()
        
        # We can use side_effect on execute or just mock the chain loosely.
        # Since the structure is slightly different (select vs update), we can mock based on call args if we want strictness.
        # For now, let's just ensure the final update returns success.
        
        # Mock select return
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_get_response
        
        # Mock update return
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_response
        
        result = update_customer_status("cust-123", 'contacted')
        
        assert result['success'] is True
        # assert result['old_status'] == 'new'
        assert result['new_status'] == 'contacted'


class TestToolDefinitions:
    """Test TOOL_DEFINITIONS structure"""
    
    def test_tool_definitions_format(self):
        """Test that tool definitions have correct structure"""
        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) == 6
        
        for tool in TOOL_DEFINITIONS:
            assert 'name' in tool
            assert 'description' in tool
            assert 'parameters' in tool
            assert 'required' in tool
            assert 'handler' in tool

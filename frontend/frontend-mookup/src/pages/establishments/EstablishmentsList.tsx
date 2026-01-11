import { useState } from 'react';
import { Plus, Filter, Search, Loader2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { establishments as mockEstablishments } from '../../data/mockData';
import { useEstablishments } from '../../api/hooks/useEstablishments';
import { config } from '../../config';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';

const EstablishmentsList = () => {
  const navigate = useNavigate();
  const openPreview = usePreviewStore((state) => state.open);
  const [searchTerm, setSearchTerm] = useState('');
  
  // API hook
  const { data: establishmentsData, isLoading } = useEstablishments();
  
  // Use mock data if enabled or API data not available
  const useMock = config.enableMockData;
  const establishments = useMock 
    ? mockEstablishments 
    : (establishmentsData?.map(e => ({
        id: e.id,
        name: e.name,
        cnpj: e.slug, // Using slug as placeholder for CNPJ
        status: e.is_active ? 'Active' : 'Inactive',
        agents: e.agent_count,
        cost: `$${e.monthly_cost.toLocaleString()}`,
      })) || mockEstablishments);
  
  // Filter by search term
  const filteredEstablishments = establishments.filter(e => 
    e.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="section">
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div className="section-title">
          <h2>Establishments</h2>
          <div style={{ display: 'flex', gap: 12 }}>
            <Button onClick={() => navigate('/establishments/new')}>
              <Plus size={14} /> New
            </Button>
            <Button variant="outline" onClick={() => openPreview('establishment-filters')}>
              <Filter size={14} /> Filters
            </Button>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <Label className="input-field" style={{ flex: 1 }}>
            <Search size={16} />
            <Input 
              placeholder="Search establishment" 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </Label>
          <Button variant="outline" onClick={() => openPreview('establishment-bulk')}>
            Bulk actions
          </Button>
        </div>
        {isLoading && !useMock ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
            <Loader2 className="animate-spin" size={32} />
          </div>
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>CNPJ</th>
                  <th>Status</th>
                  <th>Agents</th>
                  <th>Cost</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filteredEstablishments.map((establishment) => (
                  <tr key={establishment.name}>
                    <td>{establishment.name}</td>
                    <td>{establishment.cnpj}</td>
                    <td>{establishment.status}</td>
                    <td>{establishment.agents}</td>
                    <td>{establishment.cost}</td>
                    <td>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        onClick={() => navigate(`/establishments/${'id' in establishment ? establishment.id : establishment.name}`)}
                      >
                        Manage
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default EstablishmentsList;

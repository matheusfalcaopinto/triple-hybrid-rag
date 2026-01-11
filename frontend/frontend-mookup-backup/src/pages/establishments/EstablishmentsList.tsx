import { Plus, Filter, Search } from 'lucide-react';
import { establishments } from '../../data/mockData';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';

const EstablishmentsList = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div className="section-title">
          <h2>Establishments</h2>
          <div style={{ display: 'flex', gap: 12 }}>
            <Button onClick={() => openPreview('establishment-create')}>
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
            <Input placeholder="Search establishment" />
          </Label>
          <Button variant="outline" onClick={() => openPreview('establishment-bulk')}>
            Bulk actions
          </Button>
        </div>
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
              {establishments.map((establishment) => (
                <tr key={establishment.name}>
                  <td>{establishment.name}</td>
                  <td>{establishment.cnpj}</td>
                  <td>{establishment.status}</td>
                  <td>{establishment.agents}</td>
                  <td>{establishment.cost}</td>
                  <td>
                    <Button variant="ghost" size="sm" onClick={() => openPreview('establishment-manage')}>
                      Manage
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default EstablishmentsList;

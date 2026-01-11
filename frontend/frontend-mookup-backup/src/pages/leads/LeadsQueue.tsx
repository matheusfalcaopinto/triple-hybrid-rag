import { leadsColumns } from '../../data/mockData';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const cards = [
  'Solar consultation - warm lead',
  'Follow-up requested - email sent',
  'Needs financing plan',
];

const LeadsQueue = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="section-title">
        <h2>Leads pipeline</h2>
        <Button variant="secondary" size="sm" onClick={() => openPreview('leads-import-csv')}>
          Import CSV
        </Button>
      </div>
      <div className="kanban">
        {leadsColumns.map((column) => (
          <div key={column.title} className="kanban-column">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <strong>{column.title}</strong>
              <span className="tag">{column.count}</span>
            </div>
            {cards.map((card, index) => (
              <div key={index} className="card" style={{ borderRadius: 12 }}>
                {card}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default LeadsQueue;

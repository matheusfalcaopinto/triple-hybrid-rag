import { recentCalls } from '../../data/mockData';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const CallsHistory = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>Call history</h2>
          <Button variant="secondary" size="sm" onClick={() => openPreview('calls-export')}>
            Export CSV
          </Button>
        </div>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Establishment</th>
                <th>Agent</th>
                <th>Number</th>
                <th>Duration</th>
                <th>Status</th>
                <th>Tier</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {recentCalls.map((call, index) => (
                <tr key={call.id}>
                  <td>2024-08-{10 + index}</td>
                  <td>{call.establishment}</td>
                  <td>{call.agent}</td>
                  <td>{call.number}</td>
                  <td>{call.duration}</td>
                  <td>{call.status}</td>
                  <td>{call.tier}</td>
                  <td>${(index * 1.2 + 2.4).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default CallsHistory;

import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const keys = [
  { name: 'Production key', permissions: 'Calls, Agents', rateLimit: '120 rpm', lastUsed: '2m ago' },
  { name: 'Sandbox key', permissions: 'Calls (test)', rateLimit: '60 rpm', lastUsed: '1d ago' },
];

const SettingsApiKeys = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card">
        <div className="section-title">
          <h2>API keys</h2>
          <Button onClick={() => openPreview('settings-generate-key')}>Generate key</Button>
        </div>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Permissions</th>
                <th>Rate limit</th>
                <th>Last used</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.name}>
                  <td>{key.name}</td>
                  <td>{key.permissions}</td>
                  <td>{key.rateLimit}</td>
                  <td>{key.lastUsed}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default SettingsApiKeys;

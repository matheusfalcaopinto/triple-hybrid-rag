import { useState } from 'react';
import { useAPIKeys, useCreateAPIKey, useRevokeAPIKey, useToggleAPIKey } from '../../api/hooks';
import type { APIKeyScope, APIKeyCreated } from '../../api/types';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import { Switch } from '../../components/ui/switch';
import { Badge } from '../../components/ui/badge';
import { 
  Loader2, 
  Plus, 
  Trash2, 
  Copy, 
  Check, 
  Key, 
  Clock, 
  Shield,
  AlertTriangle
} from 'lucide-react';

const AVAILABLE_SCOPES: { value: APIKeyScope; label: string }[] = [
  { value: 'calls', label: 'Calls' },
  { value: 'agents', label: 'Agents' },
  { value: 'leads', label: 'Leads' },
  { value: 'campaigns', label: 'Campaigns' },
  { value: 'reports', label: 'Reports' },
  { value: 'admin', label: 'Admin' },
];

const SettingsApiKeys = () => {
  const { data: keys, isLoading, error } = useAPIKeys();
  const createKey = useCreateAPIKey();
  const revokeKey = useRevokeAPIKey();
  const toggleKey = useToggleAPIKey();

  const [isCreating, setIsCreating] = useState(false);
  const [newKeyData, setNewKeyData] = useState({
    name: '',
    scopes: [] as APIKeyScope[],
    rate_limit: 120,
    expires_in_days: 90,
  });
  const [createdKey, setCreatedKey] = useState<APIKeyCreated | null>(null);
  const [copied, setCopied] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<string | null>(null);

  const handleCreateKey = async () => {
    try {
      const result = await createKey.mutateAsync(newKeyData);
      setCreatedKey(result);
      setNewKeyData({ name: '', scopes: [], rate_limit: 120, expires_in_days: 90 });
    } catch (err) {
      console.error('Failed to create key:', err);
    }
  };

  const handleCopyKey = () => {
    if (createdKey?.key) {
      navigator.clipboard.writeText(createdKey.key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRevokeKey = async (id: string) => {
    await revokeKey.mutateAsync(id);
    setKeyToDelete(null);
  };

  const handleToggleScope = (scope: APIKeyScope) => {
    setNewKeyData((prev) => ({
      ...prev,
      scopes: prev.scopes.includes(scope)
        ? prev.scopes.filter((s) => s !== scope)
        : [...prev.scopes, scope],
    }));
  };

  const formatDate = (date?: string) => {
    if (!date) return 'Never';
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
          <Loader2 className="animate-spin" size={32} />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="card" style={{ color: 'var(--danger)', padding: 20 }}>
          Error loading API keys: {error.message}
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      {/* Created Key Modal */}
      {createdKey && (
        <div 
          style={{ 
            position: 'fixed', 
            inset: 0, 
            background: 'rgba(0,0,0,0.8)', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            zIndex: 1000 
          }}
        >
          <div className="card" style={{ maxWidth: 500, width: '90%' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <Shield size={24} style={{ color: 'var(--success)' }} />
              <h3 style={{ margin: 0 }}>API Key Created</h3>
            </div>
            <div style={{ 
              background: 'rgba(15,23,42,0.8)', 
              padding: 16, 
              borderRadius: 8, 
              marginBottom: 16,
              fontFamily: 'monospace'
            }}>
              <p style={{ margin: '0 0 8px', color: 'var(--muted)', fontSize: '0.875rem' }}>
                Copy this key now. You won't be able to see it again.
              </p>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <code style={{ 
                  flex: 1, 
                  wordBreak: 'break-all', 
                  fontSize: '0.875rem',
                  color: 'var(--success)'
                }}>
                  {createdKey.key}
                </code>
                <Button variant="secondary" size="sm" onClick={handleCopyKey}>
                  {copied ? <Check size={16} /> : <Copy size={16} />}
                </Button>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
              {createdKey.scopes.map((scope) => (
                <Badge key={scope} variant="secondary">{scope}</Badge>
              ))}
            </div>
            <Button 
              onClick={() => { setCreatedKey(null); setIsCreating(false); }}
              style={{ width: '100%' }}
            >
              Done
            </Button>
          </div>
        </div>
      )}

      {/* Delete Confirmation */}
      {keyToDelete && (
        <div 
          style={{ 
            position: 'fixed', 
            inset: 0, 
            background: 'rgba(0,0,0,0.8)', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            zIndex: 1000 
          }}
        >
          <div className="card" style={{ maxWidth: 400, width: '90%' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <AlertTriangle size={24} style={{ color: 'var(--danger)' }} />
              <h3 style={{ margin: 0 }}>Revoke API Key?</h3>
            </div>
            <p style={{ color: 'var(--muted)', marginBottom: 16 }}>
              This action cannot be undone. Any applications using this key will lose access.
            </p>
            <div style={{ display: 'flex', gap: 8 }}>
              <Button variant="secondary" onClick={() => setKeyToDelete(null)} style={{ flex: 1 }}>
                Cancel
              </Button>
              <Button 
                variant="destructive" 
                onClick={() => handleRevokeKey(keyToDelete)}
                disabled={revokeKey.isPending}
                style={{ flex: 1 }}
              >
                {revokeKey.isPending && <Loader2 className="animate-spin" size={16} style={{ marginRight: 8 }} />}
                Revoke
              </Button>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="section-title">
          <h2>API Keys</h2>
          <Button onClick={() => setIsCreating(!isCreating)}>
            <Plus size={16} style={{ marginRight: 8 }} />
            Generate Key
          </Button>
        </div>

        {/* Create Key Form */}
        {isCreating && (
          <div className="card" style={{ borderRadius: 12, marginBottom: 16 }}>
            <h3>Create New API Key</h3>
            <div style={{ display: 'grid', gap: 16 }}>
              <div>
                <Label htmlFor="keyName">Key Name</Label>
                <Input
                  id="keyName"
                  value={newKeyData.name}
                  onChange={(e) => setNewKeyData({ ...newKeyData, name: e.target.value })}
                  placeholder="e.g., Production key"
                />
              </div>
              <div>
                <Label>Permissions</Label>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                  {AVAILABLE_SCOPES.map((scope) => (
                    <Badge
                      key={scope.value}
                      variant={newKeyData.scopes.includes(scope.value) ? 'default' : 'secondary'}
                      style={{ cursor: 'pointer' }}
                      onClick={() => handleToggleScope(scope.value)}
                    >
                      {scope.label}
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="grid-2">
                <div>
                  <Label htmlFor="rateLimit">Rate Limit (requests/min)</Label>
                  <Input
                    id="rateLimit"
                    type="number"
                    value={newKeyData.rate_limit}
                    onChange={(e) => setNewKeyData({ ...newKeyData, rate_limit: parseInt(e.target.value) })}
                  />
                </div>
                <div>
                  <Label htmlFor="expiry">Expires In (days)</Label>
                  <Input
                    id="expiry"
                    type="number"
                    value={newKeyData.expires_in_days}
                    onChange={(e) => setNewKeyData({ ...newKeyData, expires_in_days: parseInt(e.target.value) })}
                  />
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <Button variant="secondary" onClick={() => setIsCreating(false)}>
                  Cancel
                </Button>
                <Button 
                  onClick={handleCreateKey}
                  disabled={!newKeyData.name || newKeyData.scopes.length === 0 || createKey.isPending}
                >
                  {createKey.isPending && <Loader2 className="animate-spin" size={16} style={{ marginRight: 8 }} />}
                  Generate Key
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Keys Table */}
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Prefix</th>
                <th>Permissions</th>
                <th>Rate Limit</th>
                <th>Last Used</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {keys?.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: 40, color: 'var(--muted)' }}>
                    <Key size={32} style={{ marginBottom: 8, opacity: 0.5 }} />
                    <p>No API keys yet. Create one to get started.</p>
                  </td>
                </tr>
              )}
              {keys?.map((key) => (
                <tr key={key.id}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Key size={16} style={{ color: 'var(--muted)' }} />
                      {key.name}
                    </div>
                  </td>
                  <td>
                    <code style={{ fontSize: '0.875rem' }}>{key.prefix}...</code>
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {key.scopes.slice(0, 3).map((scope) => (
                        <Badge key={scope} variant="secondary" style={{ fontSize: '0.75rem' }}>
                          {scope}
                        </Badge>
                      ))}
                      {key.scopes.length > 3 && (
                        <Badge variant="secondary" style={{ fontSize: '0.75rem' }}>
                          +{key.scopes.length - 3}
                        </Badge>
                      )}
                    </div>
                  </td>
                  <td>
                    <span style={{ color: 'var(--muted)' }}>{key.rate_limit} rpm</span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--muted)' }}>
                      <Clock size={14} />
                      {formatDate(key.last_used_at)}
                    </div>
                  </td>
                  <td>
                    <Switch
                      checked={key.is_active}
                      onCheckedChange={(checked) => 
                        toggleKey.mutate({ id: key.id, isActive: checked })
                      }
                    />
                  </td>
                  <td>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => setKeyToDelete(key.id)}
                    >
                      <Trash2 size={16} style={{ color: 'var(--danger)' }} />
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

export default SettingsApiKeys;

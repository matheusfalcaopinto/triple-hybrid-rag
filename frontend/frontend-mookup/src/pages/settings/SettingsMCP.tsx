import { useState, useRef, useEffect } from 'react';
import { useDashboardMetrics } from '../../api/hooks';
import type { DashboardKPI } from '../../api/types';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Badge } from '../../components/ui/badge';
import { 
  Loader2, 
  Terminal, 
  Server, 
  Play, 
  Workflow, 
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react';

interface CommandLog {
  id: string;
  command: string;
  output: string;
  status: 'success' | 'error' | 'pending';
  timestamp: Date;
}

const AVAILABLE_COMMANDS = [
  { name: 'sync-knowledge', description: 'Sync knowledge base from sources' },
  { name: 'rotate-keys', description: 'Rotate encryption/API keys' },
  { name: 'deploy-agent', description: 'Deploy agent to runtime' },
  { name: 'fetch-metrics', description: 'Fetch real-time metrics' },
  { name: 'clear-cache', description: 'Clear system caches' },
  { name: 'health-check', description: 'Run system health check' },
];

const SettingsMCP = () => {
  const { data: metrics, isLoading } = useDashboardMetrics();
  
  const [commandHistory, setCommandHistory] = useState<CommandLog[]>([]);
  const [currentCommand, setCurrentCommand] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [commandHistory]);

  const handleExecuteCommand = async (cmd?: string) => {
    const command = cmd || currentCommand;
    if (!command.trim()) return;

    const newLog: CommandLog = {
      id: Date.now().toString(),
      command,
      output: '',
      status: 'pending',
      timestamp: new Date(),
    };

    setCommandHistory((prev) => [...prev, newLog]);
    setCurrentCommand('');
    setIsExecuting(true);

    // Simulate command execution
    setTimeout(() => {
      const activeCalls = metrics?.kpis?.find((k: DashboardKPI) => k.label === 'Active Calls')?.value || 0;
      const totalAgents = metrics?.agent_utilization?.length || 0;
      
      const outputs: Record<string, { output: string; status: 'success' | 'error' }> = {
        'sync-knowledge': { output: 'Knowledge base synchronized. 42 documents updated.', status: 'success' },
        'rotate-keys': { output: 'Keys rotated successfully. New keys active.', status: 'success' },
        'deploy-agent': { output: 'Agent deployment scheduled. ETA: 30s', status: 'success' },
        'fetch-metrics': { output: `Metrics fetched: ${activeCalls} active calls, ${totalAgents} agents`, status: 'success' },
        'clear-cache': { output: 'Cache cleared: 256MB freed', status: 'success' },
        'health-check': { output: 'All systems operational. API: OK, DB: OK, Runtime: OK', status: 'success' },
      };

      const result = outputs[command] || { 
        output: `Unknown command: ${command}. Type 'help' for available commands.`, 
        status: 'error' as const 
      };

      setCommandHistory((prev) =>
        prev.map((log) =>
          log.id === newLog.id
            ? { ...log, output: result.output, status: result.status }
            : log
        )
      );
      setIsExecuting(false);
    }, 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isExecuting) {
      handleExecuteCommand();
    }
  };

  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>MCP Control Plane</h2>
          <Button onClick={() => handleExecuteCommand('health-check')}>
            <RefreshCw size={16} style={{ marginRight: 8 }} />
            Health Check
          </Button>
        </div>

        {/* Status Cards */}
        <div className="grid-2">
          <div className="card" style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <Server size={20} />
              <h3 style={{ margin: 0 }}>Server Status</h3>
            </div>
            {isLoading ? (
              <Loader2 className="animate-spin" size={20} />
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <CheckCircle size={16} style={{ color: 'var(--success)' }} />
                  <span style={{ color: 'var(--success)' }}>Online</span>
                </div>
                <p style={{ color: 'var(--muted)', margin: 0, fontSize: '0.875rem' }}>
                  {commandHistory.length} commands processed this session
                </p>
              </>
            )}
          </div>

          <div className="card" style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <Workflow size={20} />
              <h3 style={{ margin: 0 }}>Automations</h3>
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div>
                <p style={{ fontSize: '1.5rem', fontWeight: 600, margin: 0 }}>3</p>
                <p style={{ color: 'var(--muted)', margin: 0, fontSize: '0.875rem' }}>Scheduled syncs</p>
              </div>
              <div>
                <p style={{ fontSize: '1.5rem', fontWeight: 600, margin: 0 }}>1</p>
                <p style={{ color: 'var(--muted)', margin: 0, fontSize: '0.875rem' }}>Paused</p>
              </div>
            </div>
          </div>
        </div>

        {/* Available Commands */}
        <div className="card" style={{ borderRadius: 12 }}>
          <h3 style={{ marginBottom: 12 }}>Available Commands</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {AVAILABLE_COMMANDS.map((cmd) => (
              <Badge
                key={cmd.name}
                variant="secondary"
                style={{ cursor: 'pointer', padding: '8px 12px' }}
                onClick={() => handleExecuteCommand(cmd.name)}
                title={cmd.description}
              >
                <Play size={12} style={{ marginRight: 4 }} />
                {cmd.name}
              </Badge>
            ))}
          </div>
        </div>

        {/* Terminal */}
        <div className="card" style={{ borderRadius: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <Terminal size={20} />
            <h3 style={{ margin: 0 }}>Terminal</h3>
          </div>
          
          <div
            ref={terminalRef}
            style={{
              background: 'rgba(15,23,42,0.9)',
              borderRadius: 8,
              padding: 16,
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              minHeight: 200,
              maxHeight: 400,
              overflowY: 'auto',
            }}
          >
            {commandHistory.length === 0 && (
              <p style={{ color: 'var(--muted)', margin: 0 }}>
                Type a command or click one above to get started...
              </p>
            )}
            {commandHistory.map((log) => (
              <div key={log.id} style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ color: 'var(--success)' }}>$</span>
                  <span style={{ color: 'var(--info)' }}>{log.command}</span>
                  <span style={{ color: 'var(--muted)', marginLeft: 'auto', fontSize: '0.75rem' }}>
                    <Clock size={12} style={{ display: 'inline', marginRight: 4 }} />
                    {log.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                {log.status === 'pending' ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4, color: 'var(--muted)' }}>
                    <Loader2 className="animate-spin" size={14} />
                    Executing...
                  </div>
                ) : (
                  <div style={{ 
                    marginTop: 4, 
                    color: log.status === 'error' ? 'var(--danger)' : 'var(--foreground)',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 8
                  }}>
                    {log.status === 'error' ? (
                      <XCircle size={14} style={{ marginTop: 2, flexShrink: 0 }} />
                    ) : (
                      <CheckCircle size={14} style={{ marginTop: 2, flexShrink: 0, color: 'var(--success)' }} />
                    )}
                    <span>{log.output}</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <span style={{ 
                position: 'absolute', 
                left: 12, 
                top: '50%', 
                transform: 'translateY(-50%)',
                color: 'var(--success)',
                fontFamily: 'monospace'
              }}>
                $
              </span>
              <Input
                value={currentCommand}
                onChange={(e) => setCurrentCommand(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter command..."
                style={{ paddingLeft: 28, fontFamily: 'monospace' }}
                disabled={isExecuting}
              />
            </div>
            <Button onClick={() => handleExecuteCommand()} disabled={isExecuting || !currentCommand}>
              {isExecuting ? (
                <Loader2 className="animate-spin" size={16} />
              ) : (
                <Play size={16} />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsMCP;

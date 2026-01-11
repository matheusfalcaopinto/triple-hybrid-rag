import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Area } from 'recharts';
import { Play, PauseCircle, PhoneCall, AlertTriangle } from 'lucide-react';
import { StatusBadge, StatusKey } from '../../components/StatusBadge';
import { agentStatuses, kpis, recentCalls } from '../../data/mockData';
import { usePreviewStore } from '../../stores/usePreviewStore';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
} from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from '../../components/ui/table';

const activityData = [
  { name: '00h', calls: 12 },
  { name: '04h', calls: 28 },
  { name: '08h', calls: 66 },
  { name: '12h', calls: 102 },
  { name: '16h', calls: 96 },
  { name: '20h', calls: 58 },
  { name: '24h', calls: 44 },
];

const Dashboard = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="dashboard-page">
      <div className="dashboard-grid">
        {kpis.map((item) => (
          <Card key={item.label} className="dashboard-card">
            <CardHeader className="dashboard-card__header">
              <CardTitle>{item.label}</CardTitle>
              <Badge variant="info">Live</Badge>
            </CardHeader>
            <CardContent className="dashboard-card__content">
              <div className="dashboard-card__value">{item.value}</div>
              <span className="kpi-trend">
                {item.trend} · {item.trendLabel}
              </span>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="dashboard-chart">
        <CardHeader className="dashboard-card__header">
          <CardTitle>Activity (24h)</CardTitle>
          <div className="dashboard-range">
            {['24h', '7d', '30d'].map((range) => (
              <Button
                key={range}
                variant={range === '24h' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={() => openPreview('dashboard-activity-range')}
              >
                {range}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent className="dashboard-chart__canvas">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={activityData}>
              <defs>
                <linearGradient id="colorCalls" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="name" stroke="rgba(148,163,184,0.5)" />
              <YAxis stroke="rgba(148,163,184,0.5)" />
              <Tooltip cursor={{ stroke: 'rgba(99,102,241,0.4)' }} />
              <Area type="monotone" dataKey="calls" stroke="#6366f1" fill="url(#colorCalls)" />
              <Line type="monotone" dataKey="calls" stroke="#818cf8" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <div className="dashboard-columns">
        <Card className="dashboard-panel">
          <CardHeader className="dashboard-card__header">
            <CardTitle>Agent status</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => openPreview('dashboard-agent-monitor')}>
              <Play size={14} /> Monitor
            </Button>
          </CardHeader>
          <CardContent className="dashboard-agent-list">
            {agentStatuses.map((agent) => (
              <div key={agent.name} className="dashboard-agent">
                <div className="dashboard-agent__meta">
                  <div className="dashboard-agent__avatar" />
                  <div>
                    <strong>{agent.name}</strong>
                    <div>{agent.calls} calls today</div>
                  </div>
                </div>
                <StatusBadge status={agent.status as StatusKey} />
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="dashboard-panel">
          <CardHeader className="dashboard-card__header">
            <CardTitle>Alerts</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => openPreview('alerts-snooze')}>
              <PauseCircle size={14} /> Snooze
            </Button>
          </CardHeader>
          <CardContent className="dashboard-alerts">
            <div className="timeline-item">
              <strong>Budget alert</strong>
              <p>Premium tier usage trending +18% vs forecast.</p>
            </div>
            <div className="timeline-item">
              <strong>Training suggestion</strong>
              <p>Agent Hera is experiencing higher dispute rates.</p>
            </div>
            <div className="timeline-item">
              <strong>Incident</strong>
              <p>Twilio latency recovered at 04:22 UTC.</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="dashboard-table">
        <CardHeader className="dashboard-card__header">
          <CardTitle>Recent calls</CardTitle>
          <Button variant="ghost" size="sm" onClick={() => openPreview('dashboard-recent-calls')}>
            <PhoneCall size={14} /> View all
          </Button>
        </CardHeader>
        <CardContent className="table-scroll">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Establishment</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Number</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Tier</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recentCalls.map((call) => (
                <TableRow key={call.id}>
                  <TableCell>Today</TableCell>
                  <TableCell>{call.establishment}</TableCell>
                  <TableCell>{call.agent}</TableCell>
                  <TableCell>{call.number}</TableCell>
                  <TableCell>{call.duration}</TableCell>
                  <TableCell>
                    <Badge variant="success">{call.status}</Badge>
                  </TableCell>
                  <TableCell>{call.tier}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card className="dashboard-panel">
        <CardHeader className="dashboard-card__header">
          <CardTitle>Active alerts</CardTitle>
          <Badge variant="destructive">2 critical</Badge>
        </CardHeader>
        <CardContent className="dashboard-critical">
          <div className="dashboard-critical__item">
            <AlertTriangle size={18} />
            <div>
              <strong>Call drop rate</strong>
              <p>Investigate SIP trunk for APAC region - +6% drops.</p>
            </div>
          </div>
          <div className="dashboard-critical__item">
            <AlertTriangle size={18} color="var(--warning)" />
            <div>
              <strong>Knowledge base sync</strong>
              <p>3 files pending review before activation.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;

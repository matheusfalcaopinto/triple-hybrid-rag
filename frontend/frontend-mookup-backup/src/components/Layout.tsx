import {
  LayoutDashboard,
  Building2,
  Bot,
  Phone,
  Users,
  BarChart3,
  Settings,
  HelpCircle,
  Bell,
  ChevronDown,
  Moon,
  Sun,
} from 'lucide-react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { ReactNode, useEffect } from 'react';
import { useAppStore } from '../stores/useAppStore';
import { usePreviewStore } from '../stores/usePreviewStore';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Avatar } from './ui/avatar';

const navItems = [
  { label: 'Dashboard', icon: LayoutDashboard, to: '/dashboard' },
  { label: 'Establishments', icon: Building2, to: '/establishments' },
  { label: 'Agents', icon: Bot, to: '/agents' },
  { label: 'Calls', icon: Phone, to: '/calls' },
  { label: 'Leads', icon: Users, to: '/leads' },
  { label: 'Reports', icon: BarChart3, to: '/reports' },
  { label: 'Settings', icon: Settings, to: '/settings' },
  { label: 'Help', icon: HelpCircle, to: '/help' },
];

const ThemeToggle = () => {
  const theme = useAppStore((state) => state.theme);
  const toggleTheme = useAppStore((state) => state.toggleTheme);
  const Icon = theme === 'dark' ? Sun : Moon;

  return (
    <Button
      variant="outline"
      onClick={toggleTheme}
      aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      className="theme-toggle"
    >
      <Icon size={18} />
      <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
    </Button>
  );
};

const Header = ({ children, alerts }: { children: ReactNode; alerts: number }) => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="topbar">
      <div className="topbar__stack">{children}</div>
      <div className="topbar__actions">
        <Button
          variant="secondary"
          onClick={() => openPreview('notifications-center')}
          className="topbar__alerts"
        >
          <Bell size={18} />
          <span>{alerts} alerts</span>
        </Button>
        <ThemeToggle />
        <div className="topbar__profile">
          <Avatar className="topbar__avatar">TO</Avatar>
          <div>
            <div className="topbar__name">Taylor Ops</div>
            <small>Operations Lead</small>
          </div>
          <ChevronDown size={16} />
        </div>
      </div>
    </div>
  );
};

const Layout = () => {
  const location = useLocation();
  const { establishment, alerts, theme } = useAppStore();
  const breadcrumbs = location.pathname
    .split('/')
    .filter(Boolean)
    .map((segment, index, segments) => {
      const path = `/${segments.slice(0, index + 1).join('/')}`;
      return (
        <span key={path} style={{ color: 'var(--muted)' }}>
          {index > 0 ? ' / ' : ''}
          <Link to={path} style={{ color: 'inherit' }}>
            {segment.replace('-', ' ')}
          </Link>
        </span>
      );
    });

  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>
          <div className="sidebar__logo" />
          Nova Voice Cloud
        </h1>
        <nav className="nav-section">
          {navItems.map(({ label, icon: Icon, to }) => (
            <NavLink key={to} to={to} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
              <Icon size={18} />
              <span>{label}</span>
              {to === '/calls' && (
                <span className="badge" style={{ background: 'var(--error)' }} />
              )}
            </NavLink>
          ))}
        </nav>
        <div style={{ marginTop: 'auto', padding: 24, color: 'var(--muted)', fontSize: 12 }}>
          <div style={{ fontWeight: 600, color: 'var(--foreground)', marginBottom: 8 }}>
            Critical flows
          </div>
          <div className="timeline">
            <div className="timeline-item">First access onboarding</div>
            <div className="timeline-item">Agent setup guide</div>
            <div className="timeline-item">Real-time call monitor</div>
            <div className="timeline-item">Cost optimization tips</div>
          </div>
        </div>
      </aside>
      <section className="content-area">
        <Header alerts={alerts}>
          <div>
            <div className="topbar__label">Active establishment</div>
            <div className="topbar__establishment">
              <strong>{establishment}</strong>
              <Badge variant="outline">12 agents</Badge>
            </div>
          </div>
          <div className="topbar__breadcrumbs">{breadcrumbs}</div>
          <Badge variant="destructive">{alerts} alerts</Badge>
        </Header>
        <main className="main-content">
          <Outlet />
        </main>
      </section>
    </div>
  );
};

export default Layout;

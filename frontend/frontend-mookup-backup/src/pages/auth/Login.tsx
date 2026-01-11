import { ArrowRight, AudioLines, Lock, Mail, ShieldCheck } from 'lucide-react';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import './auth.css';

const Login = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="auth-shell">
      <div className="auth-grid">
        <aside className="auth-visual">
          <div className="auth-visual__overlay">
            <div className="auth-visual__brand">
              <span className="auth-logo">Pulse</span>
              <span className="auth-tag">AI Telephony Platform</span>
            </div>
            <div className="auth-visual__headline">
              <h2>Operate every conversation with confidence.</h2>
              <p>
                Unlock realtime insights, orchestrate agent workflows, and keep
                customers delighted from the first ring to the final follow-up.
              </p>
            </div>
            <ul className="auth-visual__highlights">
              <li>
                <ShieldCheck size={18} />
                Enterprise-grade security and governance
              </li>
              <li>
                <AudioLines size={18} />
                Live voice analytics with proactive alerts
              </li>
              <li>
                <ArrowRight size={18} />
                Guided setup that gets teams calling in minutes
              </li>
            </ul>
            <div className="auth-visual__footer">
              <div>
                <span className="metric-value">4.9/5</span>
                <span className="metric-label">Customer satisfaction</span>
              </div>
              <div>
                <span className="metric-value">12k+</span>
                <span className="metric-label">Daily AI-assisted calls</span>
              </div>
            </div>
          </div>
        </aside>
        <section className="auth-panel">
          <header className="auth-panel__header">
            <h1>Welcome back</h1>
            <p>Log in to orchestrate your AI-powered call operations.</p>
          </header>
          <form className="auth-form" autoComplete="off">
            <Label className="input-field">
              <Mail size={18} />
              <Input placeholder="Work email" type="email" />
            </Label>
            <Label className="input-field">
              <Lock size={18} />
              <Input placeholder="Password" type="password" />
            </Label>
            <div className="form-footer">
              <label>
                <input type="checkbox" /> Keep me signed in
              </label>
              <a className="muted-link" href="/forgot-password">
                Forgot password?
              </a>
            </div>
            <Button type="button" onClick={() => openPreview('auth-login')}>
              Sign in
            </Button>
          </form>
          <footer className="auth-panel__footer">
            <p>
              Having issues? <a href="/help/support">Contact support</a>
            </p>
          </footer>
        </section>
      </div>
    </div>
  );
};

export default Login;

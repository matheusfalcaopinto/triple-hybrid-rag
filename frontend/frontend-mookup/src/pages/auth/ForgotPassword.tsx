import { Mail, ShieldCheck, Undo2 } from 'lucide-react';
import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import './auth.css';

const ForgotPassword = () => {
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
              <h2>Resilient security baked into every workflow.</h2>
              <p>
                Enforce strong policies, trace activity, and protect sensitive
                transcripts with enterprise controls.
              </p>
            </div>
            <ul className="auth-visual__highlights">
              <li>
                <ShieldCheck size={18} />
                SOC 2 Type II and GDPR aligned practices
              </li>
              <li>
                <Undo2 size={18} />
                Instant credential recovery and audit trails
              </li>
            </ul>
            <div className="auth-visual__footer">
              <div>
                <span className="metric-value">99.99%</span>
                <span className="metric-label">Uptime across regions</span>
              </div>
              <div>
                <span className="metric-value">24/7</span>
                <span className="metric-label">Dedicated security team</span>
              </div>
            </div>
          </div>
        </aside>
        <section className="auth-panel">
          <header className="auth-panel__header">
            <h1>Reset your password</h1>
            <p>We will email you instructions to access your account again.</p>
          </header>
          <form className="auth-form" autoComplete="off">
            <Label className="input-field">
              <Mail size={18} />
              <Input placeholder="Work email" type="email" />
            </Label>
            <Button type="button" onClick={() => openPreview('auth-reset')}>
              Send reset link
            </Button>
            <p className="form-caption">
              If the email matches an account, you will receive a secure reset
              link momentarily.
            </p>
          </form>
          <footer className="auth-panel__footer">
            <p>
              Remembered your password? <a href="/login">Return to login</a>
            </p>
          </footer>
        </section>
      </div>
    </div>
  );
};

export default ForgotPassword;

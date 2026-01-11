import { usePreviewStore } from '../../stores/usePreviewStore';
import { Button } from '../../components/ui/button';

const SettingsProfile = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>Profile & preferences</h2>
          <Button variant="secondary" size="sm" onClick={() => openPreview('settings-save-profile')}>
            Save changes
          </Button>
        </div>
        <div className="grid-2">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Personal info</h3>
            <p style={{ color: 'var(--muted)' }}>Taylor Ops · ops@novavoice.ai</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Notifications</h3>
            <p style={{ color: 'var(--muted)' }}>Critical alerts · Weekly reports</p>
          </div>
        </div>
        <div className="grid-2">
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Theme</h3>
            <p style={{ color: 'var(--muted)' }}>Dark mode · Contrast boost</p>
          </div>
          <div className="card" style={{ borderRadius: 12 }}>
            <h3>Security</h3>
            <p style={{ color: 'var(--muted)' }}>2FA enabled · Last rotation Aug 12</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsProfile;

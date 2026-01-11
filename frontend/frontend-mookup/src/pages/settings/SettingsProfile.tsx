import { useState } from 'react';
import { useCurrentUser, useUpdateProfile, useChangePassword } from '../../api/hooks';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import { Switch } from '../../components/ui/switch';
import { Loader2, Save, Shield, Bell, Palette, User } from 'lucide-react';

const SettingsProfile = () => {
  const { data: user, isLoading, error } = useCurrentUser();
  const updateProfile = useUpdateProfile();
  const changePassword = useChangePassword();

  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  const [notifications, setNotifications] = useState({
    criticalAlerts: true,
    weeklyReports: true,
    callSummaries: false,
  });

  const [darkMode, setDarkMode] = useState(true);
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(true);

  // Initialize form when user data loads
  if (user && !formData.full_name && !formData.email) {
    setFormData({
      full_name: user.full_name || '',
      email: user.email || '',
    });
  }

  const handleSaveProfile = async () => {
    await updateProfile.mutateAsync(formData);
  };

  const handleChangePassword = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      alert('Passwords do not match');
      return;
    }
    await changePassword.mutateAsync({
      currentPassword: passwordData.currentPassword,
      newPassword: passwordData.newPassword,
    });
    setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
  };

  if (isLoading) {
    return (
      <div className="section">
        <div className="card" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 40 }}>
          <Loader2 className="animate-spin" size={32} />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="section">
        <div className="card" style={{ color: 'var(--danger)', padding: 20 }}>
          Error loading profile: {error.message}
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <div className="section-title">
          <h2>Profile & preferences</h2>
          <Button 
            onClick={handleSaveProfile}
            disabled={updateProfile.isPending}
          >
            {updateProfile.isPending ? (
              <Loader2 className="animate-spin" size={16} style={{ marginRight: 8 }} />
            ) : (
              <Save size={16} style={{ marginRight: 8 }} />
            )}
            Save changes
          </Button>
        </div>

        {/* Personal Info */}
        <div className="card" style={{ borderRadius: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <User size={20} />
            <h3 style={{ margin: 0 }}>Personal info</h3>
          </div>
          <div className="grid-2">
            <div>
              <Label htmlFor="full_name">Full Name</Label>
              <Input
                id="full_name"
                value={formData.full_name}
                onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
                placeholder="Enter your name"
              />
            </div>
            <div>
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                placeholder="Enter your email"
              />
            </div>
          </div>
        </div>

        {/* Notifications */}
        <div className="card" style={{ borderRadius: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <Bell size={20} />
            <h3 style={{ margin: 0 }}>Notifications</h3>
          </div>
          <div style={{ display: 'grid', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Label>Critical alerts</Label>
              <Switch 
                checked={notifications.criticalAlerts} 
                onCheckedChange={(checked) => setNotifications({ ...notifications, criticalAlerts: checked })}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Label>Weekly reports</Label>
              <Switch 
                checked={notifications.weeklyReports} 
                onCheckedChange={(checked) => setNotifications({ ...notifications, weeklyReports: checked })}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Label>Call summaries</Label>
              <Switch 
                checked={notifications.callSummaries} 
                onCheckedChange={(checked) => setNotifications({ ...notifications, callSummaries: checked })}
              />
            </div>
          </div>
        </div>

        <div className="grid-2">
          {/* Theme */}
          <div className="card" style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <Palette size={20} />
              <h3 style={{ margin: 0 }}>Theme</h3>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Label>Dark mode</Label>
              <Switch checked={darkMode} onCheckedChange={setDarkMode} />
            </div>
          </div>

          {/* Security */}
          <div className="card" style={{ borderRadius: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <Shield size={20} />
              <h3 style={{ margin: 0 }}>Security</h3>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <Label>Two-factor authentication</Label>
              <Switch checked={twoFactorEnabled} onCheckedChange={setTwoFactorEnabled} />
            </div>
            <p style={{ color: 'var(--muted)', fontSize: '0.875rem' }}>
              Last password change: {user?.updated_at ? new Date(user.updated_at).toLocaleDateString() : 'Never'}
            </p>
          </div>
        </div>

        {/* Change Password */}
        <div className="card" style={{ borderRadius: 12 }}>
          <h3 style={{ marginBottom: 16 }}>Change Password</h3>
          <div className="grid-2" style={{ gap: 12, marginBottom: 16 }}>
            <div>
              <Label htmlFor="currentPassword">Current Password</Label>
              <Input
                id="currentPassword"
                type="password"
                value={passwordData.currentPassword}
                onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
              />
            </div>
            <div></div>
            <div>
              <Label htmlFor="newPassword">New Password</Label>
              <Input
                id="newPassword"
                type="password"
                value={passwordData.newPassword}
                onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="confirmPassword">Confirm Password</Label>
              <Input
                id="confirmPassword"
                type="password"
                value={passwordData.confirmPassword}
                onChange={(e) => setPasswordData({ ...passwordData, confirmPassword: e.target.value })}
              />
            </div>
          </div>
          <Button 
            variant="secondary" 
            onClick={handleChangePassword}
            disabled={changePassword.isPending || !passwordData.currentPassword || !passwordData.newPassword}
          >
            {changePassword.isPending && <Loader2 className="animate-spin" size={16} style={{ marginRight: 8 }} />}
            Update Password
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SettingsProfile;

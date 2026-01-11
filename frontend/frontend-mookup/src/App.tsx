import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import FeaturePreviewOverlay from './components/FeaturePreviewOverlay';
import Login from './pages/auth/Login';
import ForgotPassword from './pages/auth/ForgotPassword';
import Dashboard from './pages/dashboard/Dashboard';
import EstablishmentsList from './pages/establishments/EstablishmentsList';
import EstablishmentNew from './pages/establishments/EstablishmentNew';
import EstablishmentDetails from './pages/establishments/EstablishmentDetails';
import EstablishmentBilling from './pages/establishments/EstablishmentBilling';
import AgentsList from './pages/agents/AgentsList';
import AgentCreate from './pages/agents/AgentCreate';
import AgentConfig from './pages/agents/AgentConfig';
import AgentKnowledge from './pages/agents/AgentKnowledge';
import AgentTools from './pages/agents/AgentTools';
import AgentTest from './pages/agents/AgentTest';
import CallsActive from './pages/calls/CallsActive';
import CallsHistory from './pages/calls/CallsHistory';
import CallDetails from './pages/calls/CallDetails';
import LeadsQueue from './pages/leads/LeadsQueue';
import LeadsImport from './pages/leads/LeadsImport';
import ReportsAnalytics from './pages/reports/ReportsAnalytics';
import ReportsCosts from './pages/reports/ReportsCosts';
import ReportsExport from './pages/reports/ReportsExport';
import SettingsProfile from './pages/settings/SettingsProfile';
import SettingsIntegrations from './pages/settings/SettingsIntegrations';
import SettingsApiKeys from './pages/settings/SettingsApiKeys';
import SettingsMCP from './pages/settings/SettingsMCP';
import HelpDocs from './pages/help/HelpDocs';
import HelpSupport from './pages/help/HelpSupport';

const App = () => {
  return (
    <>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />

        <Route path="/establishments">
          <Route index element={<EstablishmentsList />} />
          <Route path="new" element={<EstablishmentNew />} />
          <Route path=":id" element={<EstablishmentDetails />} />
          <Route path=":id/billing" element={<EstablishmentBilling />} />
          <Route path=":id/settings" element={<EstablishmentDetails />} />
        </Route>

        <Route path="/agents">
          <Route index element={<AgentsList />} />
          <Route path="new" element={<AgentCreate />} />
          <Route path=":id" element={<AgentConfig />} />
          <Route path=":id/config" element={<AgentConfig />} />
          <Route path=":id/knowledge" element={<AgentKnowledge />} />
          <Route path=":id/tools" element={<AgentTools />} />
          <Route path=":id/test" element={<AgentTest />} />
        </Route>

        <Route path="/calls">
          <Route index element={<CallsActive />} />
          <Route path="history" element={<CallsHistory />} />
          <Route path=":id" element={<CallDetails />} />
        </Route>

        <Route path="/leads">
          <Route index element={<LeadsQueue />} />
          <Route path="import" element={<LeadsImport />} />
        </Route>

        <Route path="/reports">
          <Route index element={<ReportsAnalytics />} />
          <Route path="analytics" element={<ReportsAnalytics />} />
          <Route path="costs" element={<ReportsCosts />} />
          <Route path="export" element={<ReportsExport />} />
        </Route>

        <Route path="/settings">
          <Route index element={<SettingsProfile />} />
          <Route path="profile" element={<SettingsProfile />} />
          <Route path="integrations" element={<SettingsIntegrations />} />
          <Route path="api-keys" element={<SettingsApiKeys />} />
          <Route path="mcp" element={<SettingsMCP />} />
        </Route>

        <Route path="/help">
          <Route index element={<HelpDocs />} />
          <Route path="docs" element={<HelpDocs />} />
          <Route path="support" element={<HelpSupport />} />
        </Route>
        </Route>
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
      <FeaturePreviewOverlay />
    </>
  );
};

export default App;

import { ReactNode } from 'react';
import {
  AlertTriangle,
  BadgeCheck,
  Bell,
  BookOpen,
  Bot,
  Building2,
  CalendarClock,
  Check,
  ChevronRight,
  ClipboardList,
  CloudDownload,
  Code,
  Download,
  FileAudio,
  FileJson,
  FileSpreadsheet,
  Filter,
  Flag,
  FolderCog,
  GitBranch,
  HelpCircle,
  Layers,
  LineChart,
  Mail,
  MessageSquare,
  Phone,
  Plus,
  ShieldCheck,
  Timer,
  Upload,
  Users,
  Waves,
  Wrench,
} from 'lucide-react';

const Pill = ({ label }: { label: string }) => <span className="preview-pill">{label}</span>;

const Stat = ({ label, value }: { label: string; value: string }) => (
  <div className="preview-stat">
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
);

const Step = ({ index, title, description }: { index: number; title: string; description: string }) => (
  <div className="preview-step">
    <div className="preview-step__index">{index}</div>
    <div>
      <h4>{title}</h4>
      <p>{description}</p>
    </div>
  </div>
);

const Card = ({
  title,
  subtitle,
  icon,
  description,
  footer,
}: {
  title: string;
  subtitle?: string;
  icon?: ReactNode;
  description?: ReactNode;
  footer?: ReactNode;
}) => (
  <div className="preview-card">
    {icon && <div className="preview-card__icon">{icon}</div>}
    <div className="preview-card__body">
      <div>
        <h4>{title}</h4>
        {subtitle && <span className="preview-card__subtitle">{subtitle}</span>}
      </div>
      {description}
    </div>
    {footer && <footer>{footer}</footer>}
  </div>
);

const TicketField = ({ label, value }: { label: string; value: string }) => (
  <div className="preview-ticket__field">
    <span>{label}</span>
    <p>{value}</p>
  </div>
);

export const featurePreviewVisuals: Record<string, ReactNode> = {
  'notifications-center': (
    <div className="preview-visual preview-visual--notifications">
      <header>
        <div>
          <h3>
            <Bell size={16} /> Live alerts
          </h3>
          <p>Escalation window: 15 min</p>
        </div>
        <Pill label="Auto-refresh" />
      </header>
      <ul>
        <li>
          <div>
            <strong>Budget overrun</strong>
            <p>Cartesia usage exceeded monthly allocation by 18%.</p>
          </div>
          <BadgeCheck size={16} />
        </li>
        <li>
          <div>
            <strong>Call quality degradation</strong>
            <p>West region MOS score dropped below 3.5.</p>
          </div>
          <AlertTriangle size={16} />
        </li>
        <li>
          <div>
            <strong>Pending review</strong>
            <p>5 calls await supervisor acknowledgement.</p>
          </div>
          <Timer size={16} />
        </li>
      </ul>
    </div>
  ),
  'auth-login': (
    <div className="preview-visual preview-visual--auth">
      <section>
        <h3>
          <ShieldCheck size={16} /> Credential check
        </h3>
        <div className="preview-form">
          <label>
            Email address
            <input placeholder="agent@acme.co" />
          </label>
          <label>
            Password
            <input type="password" placeholder="••••••••" />
          </label>
          <div className="preview-form__footer">
            <label className="preview-checkbox">
              <input type="checkbox" defaultChecked />
              <span>Remember session</span>
            </label>
            <button className="preview-button">Authenticate</button>
          </div>
        </div>
      </section>
      <aside>
        <h4>Session bootstrap</h4>
        <p>Warm up establishments, agents, and notifications in parallel.</p>
        <Stat label="Establishments" value="12" />
        <Stat label="Agents online" value="47" />
      </aside>
    </div>
  ),
  'auth-reset': (
    <div className="preview-visual preview-visual--auth">
      <section>
        <h3>
          <Mail size={16} /> Send recovery link
        </h3>
        <div className="preview-form">
          <label>
            Work email
            <input placeholder="user@brand.com" />
          </label>
          <button className="preview-button">Send email</button>
        </div>
      </section>
      <aside>
        <h4>Delivery status</h4>
        <ul className="preview-list">
          <li>
            <Check size={14} /> Link dispatched to SES
          </li>
          <li>
            <Timer size={14} /> Token expires in 15 minutes
          </li>
          <li>
            <ShieldCheck size={14} /> Rate-limited to 3 requests/hour
          </li>
        </ul>
      </aside>
    </div>
  ),
  'dashboard-activity-range': (
    <div className="preview-visual preview-visual--chart">
      <header>
        <div className="preview-pill-group">
          <Pill label="24h" />
          <Pill label="7d" />
          <Pill label="30d" />
        </div>
        <Pill label="Realtime sync" />
      </header>
      <div className="preview-chart">
        <div className="preview-chart__line" />
        <div className="preview-chart__line preview-chart__line--secondary" />
        <div className="preview-chart__axis" />
      </div>
      <footer>
        <Stat label="Successful calls" value="1.2k" />
        <Stat label="Sentiment uplift" value="+12%" />
      </footer>
    </div>
  ),
  'dashboard-agent-monitor': (
    <div className="preview-visual preview-visual--monitor">
      <section>
        <header>
          <Waves size={16} /> Live waveform
        </header>
        <div className="preview-waveform">
          <div />
          <div />
          <div />
        </div>
        <footer>
          <Pill label="Sentiment: Positive" />
          <Pill label="Duration: 04:21" />
        </footer>
      </section>
      <aside>
        <h4>Supervisor tools</h4>
        <button className="preview-button preview-button--ghost">
          <MessageSquare size={14} /> Whisper
        </button>
        <button className="preview-button preview-button--danger">
          <AlertTriangle size={14} /> Intervene
        </button>
        <div className="preview-transcript">
          <p>
            <span className="preview-transcript__speaker">Agent</span> Hi Sarah, thanks for calling Nimbus Energy.
          </p>
          <p>
            <span className="preview-transcript__speaker">Customer</span> We received an outage alert—can you check status?
          </p>
          <p>
            <span className="preview-transcript__speaker">Agent</span> I see a localized disruption in Austin resolved 2 mins ago.
          </p>
        </div>
      </aside>
    </div>
  ),
  'alerts-snooze': (
    <div className="preview-visual preview-visual--modal">
      <header>
        <h3>
          <Timer size={16} /> Snooze alert
        </h3>
        <Pill label="Billing spike" />
      </header>
      <div className="preview-modal__body">
        <div className="preview-radio-group">
          <label>
            <input type="radio" name="snooze" defaultChecked /> 1 hour
          </label>
          <label>
            <input type="radio" name="snooze" /> Until tomorrow
          </label>
          <label>
            <input type="radio" name="snooze" /> Custom
          </label>
        </div>
        <textarea placeholder="Add context for your team" />
      </div>
      <footer>
        <button className="preview-button preview-button--ghost">Cancel</button>
        <button className="preview-button">Confirm snooze</button>
      </footer>
    </div>
  ),
  'dashboard-recent-calls': (
    <div className="preview-visual preview-visual--table">
      <header>
        <h3>
          <Phone size={16} /> Recent calls
        </h3>
        <Pill label="Last 10" />
      </header>
      <table>
        <thead>
          <tr>
            <th>Agent</th>
            <th>Number</th>
            <th>Tier</th>
            <th>Status</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Robin Singh</td>
            <td>(415) 555-0192</td>
            <td>Premium</td>
            <td>
              <span className="preview-badge preview-badge--success">Completed</span>
            </td>
            <td>$2.47</td>
          </tr>
          <tr>
            <td>Harper Lee</td>
            <td>(213) 555-2441</td>
            <td>Economic</td>
            <td>
              <span className="preview-badge preview-badge--warning">Follow-up</span>
            </td>
            <td>$0.88</td>
          </tr>
          <tr>
            <td>Milo Chavez</td>
            <td>(628) 555-1223</td>
            <td>Premium</td>
            <td>
              <span className="preview-badge preview-badge--danger">Escalated</span>
            </td>
            <td>$3.74</td>
          </tr>
        </tbody>
      </table>
    </div>
  ),
  'establishment-create': (
    <div className="preview-visual preview-visual--wizard">
      <Step index={1} title="Basic data" description="Capture CNPJ, legal name, and HQ location." />
      <Step index={2} title="Contacts" description="Add finance, operations, and compliance leads." />
      <Step index={3} title="Config" description="Select establishment tier, hours, and timezone." />
      <Step index={4} title="Review" description="Confirm SLA, budget, and launch window." />
    </div>
  ),
  'establishment-filters': (
    <div className="preview-visual preview-visual--filters">
      <h3>
        <Filter size={16} /> Filter establishments
      </h3>
      <div className="preview-filter-group">
        <label>
          Status
          <select>
            <option>All</option>
            <option>Active</option>
            <option>Paused</option>
            <option>Pending</option>
          </select>
        </label>
        <label>
          Region
          <select>
            <option>Global</option>
            <option>North America</option>
            <option>LATAM</option>
            <option>EMEA</option>
          </select>
        </label>
      </div>
      <div className="preview-pill-group">
        <Pill label="> 50 agents" />
        <Pill label="Premium tier" />
        <Pill label="Budget alerts" />
      </div>
    </div>
  ),
  'establishment-bulk': (
    <div className="preview-visual preview-visual--modal">
      <header>
        <h3>
          <ClipboardList size={16} /> Bulk actions
        </h3>
      </header>
      <div className="preview-modal__body">
        <label className="preview-checkbox">
          <input type="checkbox" defaultChecked /> Notify account owners
        </label>
        <label className="preview-checkbox">
          <input type="checkbox" /> Adjust call caps
        </label>
        <label className="preview-checkbox">
          <input type="checkbox" /> Tag as priority
        </label>
      </div>
      <footer>
        <button className="preview-button preview-button--ghost">Cancel</button>
        <button className="preview-button">Apply to 8 establishments</button>
      </footer>
    </div>
  ),
  'establishment-manage': (
    <div className="preview-visual preview-visual--actions">
      <Card
        title="Operational controls"
        icon={<Wrench size={16} />}
        description={
          <ul className="preview-list">
            <li>Pause/resume inbound flows</li>
            <li>Override IVR entry point</li>
            <li>Route escalations to standby agent</li>
          </ul>
        }
        footer={<button className="preview-button">Open control panel</button>}
      />
      <Card
        title="Insights"
        icon={<LineChart size={16} />}
        description={<p>View handle time, transfer rate, and CSAT vs. benchmarks.</p>}
        footer={<button className="preview-button preview-button--ghost">View analytics</button>}
      />
    </div>
  ),
  'establishment-activation-pdf': (
    <div className="preview-visual preview-visual--document">
      <header>
        <FileSpreadsheet size={16} /> Launch readiness checklist
      </header>
      <ul>
        <li>
          <Check size={14} /> Twilio number pool provisioned
        </li>
        <li>
          <Check size={14} /> Voice prompts reviewed by legal
        </li>
        <li>
          <Check size={14} /> Budget guardrails confirmed
        </li>
        <li>
          <Timer size={14} /> Pending: MCP automation smoke test
        </li>
      </ul>
      <footer>
        <button className="preview-button">Export PDF</button>
      </footer>
    </div>
  ),
  'establishment-tab-overview': (
    <div className="preview-visual preview-visual--overview">
      <div className="preview-overview__header">
        <h3>
          <Building2 size={16} /> Overview snapshot
        </h3>
        <Pill label="Active" />
      </div>
      <div className="preview-overview__grid">
        <Stat label="Agents" value="34" />
        <Stat label="Monthly minutes" value="92k" />
        <Stat label="CSAT" value="4.7" />
      </div>
      <p>Surface health indicators, SLAs, and automation coverage.</p>
    </div>
  ),
  'establishment-tab-agents': (
    <div className="preview-visual preview-visual--agents">
      <header>
        <Users size={16} /> Agent roster
      </header>
      <div className="preview-agent-grid">
        <Card
          title="Eva Duarte"
          subtitle="Premium · Online"
          icon={<Bot size={16} />}
          description={<p>Energy outage concierge</p>}
        />
        <Card
          title="Noah James"
          subtitle="Economic · Paused"
          icon={<Bot size={16} />}
          description={<p>Billing FAQs specialist</p>}
        />
        <Card
          title="Zara Lee"
          subtitle="Premium · Calling"
          icon={<Bot size={16} />}
          description={<p>Enterprise renewals</p>}
        />
      </div>
    </div>
  ),
  'establishment-tab-billing': (
    <div className="preview-visual preview-visual--billing">
      <header>
        <FileSpreadsheet size={16} /> Billing timeline
      </header>
      <div className="preview-pill-group">
        <Pill label="Usage" />
        <Pill label="Invoices" />
        <Pill label="Budget" />
      </div>
      <div className="preview-chart preview-chart--bar">
        <div className="preview-bar" style={{ height: '70%' }} />
        <div className="preview-bar" style={{ height: '50%' }} />
        <div className="preview-bar" style={{ height: '90%' }} />
        <div className="preview-bar" style={{ height: '65%' }} />
      </div>
    </div>
  ),
  'establishment-tab-settings': (
    <div className="preview-visual preview-visual--settings">
      <header>
        <FolderCog size={16} /> Configurations
      </header>
      <ul className="preview-list">
        <li>Business hours · Mon–Sun 6am–11pm</li>
        <li>Routing strategy · Skills-based</li>
        <li>Notification contacts · Ops + Finance</li>
      </ul>
      <button className="preview-button">Edit settings</button>
    </div>
  ),
  'establishment-add-agent': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <Plus size={16} /> Add establishment agent
      </h3>
      <div className="preview-form">
        <label>
          Agent name
          <input placeholder="Night Shift Concierge" />
        </label>
        <label>
          Voice tier
          <select>
            <option>Premium ($0.15)</option>
            <option>Economic ($0.12)</option>
          </select>
        </label>
        <label>
          Persona summary
          <textarea placeholder="Describe the scope and personality" />
        </label>
      </div>
      <button className="preview-button">Create agent</button>
    </div>
  ),
  'establishment-billing-download': (
    <div className="preview-visual preview-visual--table">
      <header>
        <h3>
          <Download size={16} /> Invoice center
        </h3>
        <Pill label="FY24" />
      </header>
      <table>
        <thead>
          <tr>
            <th>Month</th>
            <th>Total</th>
            <th>Status</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>April</td>
            <td>$42,180</td>
            <td>Paid</td>
            <td>
              <button className="preview-button preview-button--ghost">Download</button>
            </td>
          </tr>
          <tr>
            <td>May</td>
            <td>$45,902</td>
            <td>Processing</td>
            <td>
              <button className="preview-button preview-button--ghost">Download</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  ),
  'establishment-budget-config': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <Flag size={16} /> Budget guardrails
      </h3>
      <div className="preview-form">
        <label>
          Monthly cap
          <input defaultValue="$50,000" />
        </label>
        <label>
          Alert threshold
          <input defaultValue="85%" />
        </label>
        <label>
          Escalation channel
          <select>
            <option>Slack</option>
            <option>Email</option>
            <option>PagerDuty</option>
          </select>
        </label>
      </div>
      <footer>
        <button className="preview-button preview-button--ghost">Cancel</button>
        <button className="preview-button">Save guardrails</button>
      </footer>
    </div>
  ),
  'agents-new': (
    <div className="preview-visual preview-visual--wizard">
      <Step index={1} title="Basics" description="Name, persona, and establishment assignment." />
      <Step index={2} title="Voice" description="Select tier, language, and tone." />
      <Step index={3} title="Prompt" description="Seed instructions with templates." />
      <Step index={4} title="Deploy" description="Choose channels and activation window." />
    </div>
  ),
  'agents-filters': (
    <div className="preview-visual preview-visual--filters">
      <h3>
        <Filter size={16} /> Agent filters
      </h3>
      <div className="preview-filter-group">
        <label>
          Status
          <select>
            <option>All agents</option>
            <option>Online</option>
            <option>Calling</option>
            <option>Offline</option>
          </select>
        </label>
        <label>
          Tier
          <select>
            <option>All tiers</option>
            <option>Premium</option>
            <option>Economic</option>
          </select>
        </label>
      </div>
      <div className="preview-pill-group">
        <Pill label="Language: Portuguese" />
        <Pill label="CSAT > 4.5" />
      </div>
    </div>
  ),
  'agents-configure': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <Bot size={16} /> Configure voice agent
      </h3>
      <div className="preview-form">
        <label>
          Greeting
          <input defaultValue="Olá! Estou aqui para ajudar com sua conta." />
        </label>
        <label>
          Max call duration
          <input defaultValue="12 minutes" />
        </label>
        <label>
          Escalation rule
          <select>
            <option>Transfer to supervisor</option>
            <option>Schedule callback</option>
          </select>
        </label>
      </div>
      <button className="preview-button">Publish updates</button>
    </div>
  ),
  'knowledge-levels': (
    <div className="preview-visual preview-visual--knowledge">
      <aside>
        <h4>Knowledge layers</h4>
        <ul>
          <li className="active">General</li>
          <li>Establishment</li>
          <li>Agent-specific</li>
        </ul>
      </aside>
      <section>
        <header>
          <Layers size={16} /> Curated intents
        </header>
        <div className="preview-pill-group">
          <Pill label="Billing" />
          <Pill label="Technical" />
          <Pill label="Collections" />
        </div>
        <p>Control fallback order and override collisions between datasets.</p>
      </section>
    </div>
  ),
  'knowledge-upload': (
    <div className="preview-visual preview-visual--upload">
      <div className="preview-upload__dropzone">
        <Upload size={24} />
        <p>Drop playbooks, FAQs, or scripts</p>
        <span>Supported: PDF, DOCX, HTML</span>
      </div>
      <div className="preview-upload__queue">
        <header>Processing queue</header>
        <ul>
          <li>
            <span>Escalation SOP.pdf</span>
            <span className="preview-badge preview-badge--success">Indexed</span>
          </li>
          <li>
            <span>New plans 2025.docx</span>
            <span className="preview-badge preview-badge--warning">Parsing</span>
          </li>
        </ul>
      </div>
    </div>
  ),
  'agent-integration-add': (
    <div className="preview-visual preview-visual--integrations">
      <Card
        title="Google Calendar"
        subtitle="Sync meetings"
        icon={<CalendarClock size={16} />}
        description={<p>Auto-schedule callbacks with agent availability.</p>}
        footer={<button className="preview-button">Connect</button>}
      />
      <Card
        title="Zendesk"
        subtitle="Ticket sync"
        icon={<HelpCircle size={16} />}
        description={<p>Log conversation summaries and dispositions.</p>}
        footer={<button className="preview-button">Connect</button>}
      />
      <Card
        title="Salesforce"
        subtitle="Pipeline updates"
        icon={<GitBranch size={16} />}
        description={<p>Push call outcomes into CRM opportunities.</p>}
        footer={<button className="preview-button">Connect</button>}
      />
    </div>
  ),
  'agent-integration-configure': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <FolderCog size={16} /> Integration settings
      </h3>
      <div className="preview-form">
        <label>
          Target object
          <select>
            <option>Zendesk ticket</option>
            <option>Salesforce task</option>
          </select>
        </label>
        <label>
          Trigger event
          <select>
            <option>Call completed</option>
            <option>Call escalated</option>
          </select>
        </label>
        <label>
          Field mapping
          <textarea placeholder="Map transcript summary → description" />
        </label>
      </div>
      <button className="preview-button">Save configuration</button>
    </div>
  ),
  'agent-mcp-builder': (
    <div className="preview-visual preview-visual--builder">
      <header>
        <Code size={16} /> MCP automation
      </header>
      <div className="preview-builder__canvas">
        <div className="preview-node">
          <h4>Webhook trigger</h4>
          <p>call.completed</p>
        </div>
        <ChevronRight size={18} />
        <div className="preview-node">
          <h4>Transform</h4>
          <p>Summarize transcript</p>
        </div>
        <ChevronRight size={18} />
        <div className="preview-node">
          <h4>Action</h4>
          <p>Post to Slack</p>
        </div>
      </div>
      <footer>
        <button className="preview-button">Preview run</button>
      </footer>
    </div>
  ),
  'agent-test-download': (
    <div className="preview-visual preview-visual--code">
      <header>
        <FileJson size={16} /> Debug export
      </header>
      <pre>{`{
  "callId": "call_92481",
  "agent": "Renewals Concierge",
  "actions": [
    { "type": "say", "content": "Welcome back!" },
    { "type": "gather", "field": "accountNumber" }
  ]
}`}</pre>
      <button className="preview-button">Download JSON</button>
    </div>
  ),
  'calls-intervene': (
    <div className="preview-visual preview-visual--modal">
      <header>
        <h3>
          <AlertTriangle size={16} /> Intervene call
        </h3>
        <Pill label="Customer risk" />
      </header>
      <div className="preview-modal__body">
        <p>Connect a supervisor line-in and notify compliance.</p>
        <label className="preview-checkbox">
          <input type="checkbox" defaultChecked /> Record supervisor segment
        </label>
        <label className="preview-checkbox">
          <input type="checkbox" /> Send recap to Slack #war-room
        </label>
      </div>
      <footer>
        <button className="preview-button preview-button--ghost">Cancel</button>
        <button className="preview-button preview-button--danger">Confirm intervene</button>
      </footer>
    </div>
  ),
  'calls-export': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <CloudDownload size={16} /> Export call history
      </h3>
      <div className="preview-form">
        <label>
          Format
          <select>
            <option>CSV</option>
            <option>Parquet</option>
            <option>JSON</option>
          </select>
        </label>
        <label>
          Date range
          <input defaultValue="Last 30 days" />
        </label>
        <label>
          Columns
          <textarea defaultValue="callId, agent, duration, cost, sentiment" />
        </label>
      </div>
      <button className="preview-button">Generate export</button>
    </div>
  ),
  'call-download-audio': (
    <div className="preview-visual preview-visual--audio">
      <header>
        <FileAudio size={16} /> Audio download
      </header>
      <div className="preview-audio">
        <div className="preview-waveform">
          <div />
          <div />
          <div />
        </div>
        <div className="preview-audio__meta">
          <Stat label="Duration" value="08:14" />
          <Stat label="Format" value="WAV" />
          <Stat label="Storage" value="S3" />
        </div>
      </div>
      <footer>
        <button className="preview-button">Download WAV</button>
        <button className="preview-button preview-button--ghost">Request transcript</button>
      </footer>
    </div>
  ),
  'leads-import-csv': (
    <div className="preview-visual preview-visual--wizard">
      <Step index={1} title="Upload" description="Drop CSV with lead metadata." />
      <Step index={2} title="Map" description="Align columns to phone, email, owner." />
      <Step index={3} title="Enrich" description="Validate consent and deduplicate." />
      <Step index={4} title="Confirm" description="Preview segments and launch calls." />
    </div>
  ),
  'leads-download-report': (
    <div className="preview-visual preview-visual--report">
      <header>
        <FileSpreadsheet size={16} /> Import diagnostics
      </header>
      <ul>
        <li>
          <span>Processed</span>
          <strong>5,000</strong>
        </li>
        <li>
          <span>Valid numbers</span>
          <strong>4,612</strong>
        </li>
        <li>
          <span>Duplicates</span>
          <strong>188</strong>
        </li>
      </ul>
      <button className="preview-button">Download CSV</button>
    </div>
  ),
  'reports-view-invoices': (
    <div className="preview-visual preview-visual--chart">
      <header>
        <h3>
          <LineChart size={16} /> Invoice trend
        </h3>
      </header>
      <div className="preview-chart preview-chart--area">
        <div className="preview-area" />
        <div className="preview-axis preview-axis--x" />
        <div className="preview-axis preview-axis--y" />
      </div>
      <footer>
        <Stat label="QTD spend" value="$132k" />
        <Stat label="Variance" value="-8%" />
      </footer>
    </div>
  ),
  'reports-create-template': (
    <div className="preview-visual preview-visual--builder">
      <header>
        <ClipboardList size={16} /> Export template builder
      </header>
      <div className="preview-template">
        <div>
          <h4>Available fields</h4>
          <ul>
            <li>Call duration</li>
            <li>Agent tier</li>
            <li>Customer sentiment</li>
            <li>Disposition</li>
          </ul>
        </div>
        <div>
          <h4>Selected columns</h4>
          <ul>
            <li>Call ID</li>
            <li>Agent name</li>
            <li>Cost</li>
          </ul>
        </div>
        <div>
          <h4>Preview</h4>
          <p>call_9812 · Eva Duarte · $2.14</p>
        </div>
      </div>
    </div>
  ),
  'reports-schedule-export': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <CalendarClock size={16} /> Schedule export
      </h3>
      <div className="preview-form">
        <label>
          Frequency
          <select>
            <option>Weekly</option>
            <option>Monthly</option>
            <option>Quarterly</option>
          </select>
        </label>
        <label>
          Delivery channel
          <select>
            <option>Email</option>
            <option>S3</option>
            <option>Webhook</option>
          </select>
        </label>
        <label>
          Recipients
          <input defaultValue="finance@acme.co" />
        </label>
      </div>
      <button className="preview-button">Schedule</button>
    </div>
  ),
  'settings-save-profile': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <Users size={16} /> Profile preferences
      </h3>
      <div className="preview-form">
        <label>
          Display name
          <input defaultValue="Jordan Matthews" />
        </label>
        <label>
          Theme
          <select>
            <option>Light</option>
            <option>Dark</option>
          </select>
        </label>
        <label className="preview-checkbox">
          <input type="checkbox" defaultChecked /> Enable MFA prompts
        </label>
      </div>
      <button className="preview-button">Save profile</button>
    </div>
  ),
  'settings-connect-integration': (
    <div className="preview-visual preview-visual--integrations">
      <Card
        title="Twilio"
        subtitle="Telephony"
        icon={<Phone size={16} />}
        description={<p>Manage phone numbers and SIP trunks.</p>}
        footer={<button className="preview-button">Connect</button>}
      />
      <Card
        title="Slack"
        subtitle="Collaboration"
        icon={<MessageSquare size={16} />}
        description={<p>Send alerts and share call recaps in channels.</p>}
        footer={<button className="preview-button">Connect</button>}
      />
      <Card
        title="PagerDuty"
        subtitle="On-call"
        icon={<AlertTriangle size={16} />}
        description={<p>Trigger incidents for service degradations.</p>}
        footer={<button className="preview-button">Connect</button>}
      />
    </div>
  ),
  'settings-manage-integration': (
    <div className="preview-visual preview-visual--form">
      <h3>
        <FolderCog size={16} /> Manage integration
      </h3>
      <div className="preview-form">
        <label>
          Status
          <select>
            <option>Active</option>
            <option>Paused</option>
          </select>
        </label>
        <label>
          API key
          <input defaultValue="pd_live_••••" />
        </label>
        <label className="preview-checkbox">
          <input type="checkbox" defaultChecked /> Send weekend alerts
        </label>
      </div>
      <button className="preview-button">Update integration</button>
    </div>
  ),
  'settings-generate-key': (
    <div className="preview-visual preview-visual--code">
      <header>
        <ShieldCheck size={16} /> Generate API key
      </header>
      <div className="preview-api-key">
        <p>Scope</p>
        <div className="preview-pill-group">
          <Pill label="Calls:read" />
          <Pill label="Agents:write" />
        </div>
        <button className="preview-button">Create new key</button>
        <code>ps_live_1b4e4x92</code>
      </div>
    </div>
  ),
  'settings-open-terminal': (
    <div className="preview-visual preview-visual--terminal">
      <header>
        <Code size={16} /> MCP terminal
      </header>
      <pre>{`$ connect mcp-production
✔ Authenticated as ops@acme.co
$ list automations
• standby_retrain
• budget_guardrail
• sentiment_qa`}</pre>
      <footer>
        <button className="preview-button">Open secure session</button>
      </footer>
    </div>
  ),
  'help-docs-category': (
    <div className="preview-visual preview-visual--docs">
      <aside>
        <h4>Categories</h4>
        <ul>
          <li className="active">Voice AI basics</li>
          <li>Deploying agents</li>
          <li>Billing</li>
          <li>Compliance</li>
        </ul>
      </aside>
      <section>
        <header>
          <BookOpen size={16} /> Voice AI basics
        </header>
        <p>Introduce teams to agent orchestration, fallback logic, and analytics.</p>
        <button className="preview-button">Open guide</button>
      </section>
    </div>
  ),
  'help-submit-ticket': (
    <div className="preview-visual preview-visual--ticket">
      <header>
        <MessageSquare size={16} /> Submit support ticket
      </header>
      <div className="preview-ticket">
        <TicketField label="Priority" value="High" />
        <TicketField label="Category" value="Realtime monitoring" />
        <TicketField label="Summary" value="Supervisors unable to join live calls." />
      </div>
      <button className="preview-button">Send to support</button>
    </div>
  ),
};

export const getFeaturePreviewVisual = (id: string) => featurePreviewVisuals[id] ?? null;

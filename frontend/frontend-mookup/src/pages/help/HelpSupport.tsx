import { usePreviewStore } from '../../stores/usePreviewStore';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Textarea } from '../../components/ui/textarea';

const faqs = [
  { question: 'How do I activate premium voice tiers?', answer: 'Navigate to Agents > Voice Config and toggle Premium. Ensure billing has capacity.' },
  { question: 'Can I import leads automatically?', answer: 'Yes, connect your CRM in Integrations or schedule S3 ingests via MCP.' },
  { question: 'Where are call recordings stored?', answer: 'Recordings are encrypted and stored in your configured cloud bucket with retention policies.' },
];

const HelpSupport = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="section">
      <Card className="support-card">
        <CardHeader className="section-title">
          <CardTitle>Support</CardTitle>
          <Button variant="secondary" size="sm" onClick={() => openPreview('help-submit-ticket')}>
            Submit ticket
          </Button>
        </CardHeader>
        <CardContent className="support-content">
          <div className="support-faqs">
            {faqs.map((faq) => (
              <Card key={faq.question} className="support-faq">
                <CardHeader>
                  <CardTitle>{faq.question}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p>{faq.answer}</p>
                </CardContent>
              </Card>
            ))}
          </div>
          <Card className="support-contact">
            <CardHeader>
              <CardTitle>Contact support</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="support-meta">support@novavoice.ai · SLA 2 hours</p>
              <Textarea placeholder="Describe your issue" />
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};

export default HelpSupport;

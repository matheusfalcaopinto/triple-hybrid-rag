import { usePreviewStore } from '../../stores/usePreviewStore';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '../../components/ui/card';
import { Button } from '../../components/ui/button';

const categories = ['Getting started', 'Voice tiers', 'Knowledge base', 'Compliance'];

const HelpDocs = () => {
  const openPreview = usePreviewStore((state) => state.open);

  return (
    <div className="help-docs">
      <Card className="help-docs__nav">
        <CardHeader>
          <CardTitle>Documentation</CardTitle>
        </CardHeader>
        <CardContent className="help-docs__list">
          {categories.map((category, index) => (
            <Button
              key={category}
              variant={index === 0 ? 'secondary' : 'ghost'}
              className="help-docs__tab"
              onClick={() => openPreview('help-docs-category')}
            >
              {category}
            </Button>
          ))}
        </CardContent>
      </Card>
      <Card className="help-docs__article">
        <CardHeader>
          <CardTitle>Building your first AI agent</CardTitle>
        </CardHeader>
        <CardContent>
          <article>
            <p>
              Agents represent specialized AI personas tuned for your establishment
              needs. Start with a clear objective, select the appropriate voice
              tier, and map the knowledge base assets required for accurate
              responses.
            </p>
            <p>
              Use the onboarding wizard to configure call routing, schedule, and
              escalation rules. Once published, monitor performance from the
              dashboard and iterate on prompts using the test console.
            </p>
          </article>
        </CardContent>
      </Card>
    </div>
  );
};

export default HelpDocs;

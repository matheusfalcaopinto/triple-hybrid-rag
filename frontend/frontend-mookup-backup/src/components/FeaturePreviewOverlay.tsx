import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X, Lightbulb, CheckCircle2 } from 'lucide-react';
import { usePreviewStore } from '../stores/usePreviewStore';
import { featurePreviews } from '../data/featurePreviews';
import { getFeaturePreviewVisual } from './FeaturePreviewVisuals';
import { Button } from './ui/button';

const FeaturePreviewOverlay = () => {
  const { current, close } = usePreviewStore();
  const preview = current ? featurePreviews[current] ?? featurePreviews.default : null;
  const visual = current ? getFeaturePreviewVisual(current) : null;

  useEffect(() => {
    if (!current) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        close();
      }
    };

    document.addEventListener('keydown', onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [current, close]);

  if (!preview) {
    return null;
  }

  return createPortal(
    <div className="feature-preview__backdrop" role="presentation" onClick={close}>
      <div
        role="dialog"
        aria-modal="true"
        className="feature-preview__container"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="feature-preview__header">
          <div>
            <span className="feature-preview__eyebrow">Blueprint preview</span>
            <h2>{preview.title}</h2>
            <p>{preview.subtitle}</p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="feature-preview__close"
            onClick={close}
            aria-label="Close preview"
          >
            <X size={18} />
          </Button>
        </header>

        <section className="feature-preview__body">
          <div className="feature-preview__summary">
            <aside>
              <div className="feature-preview__callout">
                <Lightbulb size={18} />
                <p>{preview.description}</p>
              </div>
              <ul>
                {preview.highlights.map((item) => (
                  <li key={item}>
                    <CheckCircle2 size={16} />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </aside>
            <div className="feature-preview__sections">
              {preview.sections.map((section) => (
                <div key={section.title} className="feature-preview__section">
                  <h3>{section.title}</h3>
                  <ol>
                    {section.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ol>
                </div>
              ))}
            </div>
          </div>
          {visual && <div className="feature-preview__visual">{visual}</div>}
        </section>

        <footer className="feature-preview__footer">
          <Button onClick={close}>{preview.ctaLabel ?? 'Close'}</Button>
        </footer>
      </div>
    </div>,
    document.body,
  );
};

export default FeaturePreviewOverlay;

import { Circle, CircleDot } from 'lucide-react';

const colors: Record<string, string> = {
  online: 'var(--success)',
  offline: 'var(--muted)',
  calling: 'var(--info)',
  error: 'var(--error)',
  paused: 'var(--warning)',
};

export type StatusKey = keyof typeof colors;

interface StatusBadgeProps {
  status: StatusKey;
  label?: string;
}

export const StatusBadge = ({ status, label }: StatusBadgeProps) => {
  const color = colors[status];
  const Icon = status === 'calling' ? CircleDot : Circle;

  return (
    <span className="badge-status" style={{ color }}>
      <Icon size={12} style={{ color }} />
      {label ?? status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};

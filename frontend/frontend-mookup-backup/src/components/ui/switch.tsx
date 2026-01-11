import * as React from 'react';
import { cn } from '../../lib/utils';

export interface SwitchProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onChange'> {
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
}

const Switch = React.forwardRef<HTMLButtonElement, SwitchProps>(
  ({ className, checked = false, onCheckedChange, ...props }, ref) => {
    return (
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        className={cn('ui-switch', checked && 'ui-switch--checked', className)}
        onClick={() => onCheckedChange?.(!checked)}
        ref={ref}
        {...props}
      >
        <span className="ui-switch__thumb" />
      </button>
    );
  }
);
Switch.displayName = 'Switch';

export { Switch };

import * as React from 'react';
import { cn } from '../../lib/utils';

interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {
  src?: string;
  alt?: string;
}

const Avatar = React.forwardRef<HTMLDivElement, AvatarProps>(
  ({ className, src, alt, children, ...props }, ref) => {
    return (
      <div ref={ref} className={cn('ui-avatar', className)} {...props}>
        {src ? <img src={src} alt={alt} /> : children}
      </div>
    );
  }
);
Avatar.displayName = 'Avatar';

export { Avatar };

import * as React from 'react';
import { cn } from '../../lib/utils';

type TabsContextValue = {
  value: string;
  onValueChange?: (value: string) => void;
};

const TabsContext = React.createContext<TabsContextValue | undefined>(undefined);

interface TabsProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
  onValueChange?: (value: string) => void;
}

const Tabs = React.forwardRef<HTMLDivElement, TabsProps>(
  ({ className, children, value, onValueChange, ...props }, ref) => {
    return (
      <TabsContext.Provider value={{ value, onValueChange }}>
        <div ref={ref} className={cn('ui-tabs', className)} {...props}>
          {children}
        </div>
      </TabsContext.Provider>
    );
  }
);
Tabs.displayName = 'Tabs';

const TabsList = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('ui-tabs__list', className)} {...props} />
  )
);
TabsList.displayName = 'TabsList';

interface TabsTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string;
}

const TabsTrigger = React.forwardRef<HTMLButtonElement, TabsTriggerProps>(
  ({ className, value, ...props }, ref) => {
    const context = React.useContext(TabsContext);
    if (!context) {
      throw new Error('TabsTrigger must be used within Tabs');
    }

    const isActive = context.value === value;

    return (
      <button
        type="button"
        className={cn('ui-tabs__trigger', isActive && 'ui-tabs__trigger--active', className)}
        onClick={() => context.onValueChange?.(value)}
        aria-selected={isActive}
        ref={ref}
        {...props}
      />
    );
  }
);
TabsTrigger.displayName = 'TabsTrigger';

interface TabsContentProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
}

const TabsContent = React.forwardRef<HTMLDivElement, TabsContentProps>(
  ({ className, value, ...props }, ref) => {
    const context = React.useContext(TabsContext);
    if (!context) {
      throw new Error('TabsContent must be used within Tabs');
    }

    if (context.value !== value) {
      return null;
    }

    return <div ref={ref} className={cn('ui-tabs__content', className)} {...props} />;
  }
);
TabsContent.displayName = 'TabsContent';

export { Tabs, TabsList, TabsTrigger, TabsContent };

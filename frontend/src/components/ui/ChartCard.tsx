import { cn } from "../../lib/utils";

interface SkeletonProps { className?: string }

export function Skeleton({ className }: SkeletonProps) {
  return <div className={cn("animate-pulse rounded bg-gray-200", className)} />;
}

interface CardProps {
  title?: string;
  loading?: boolean;
  children: React.ReactNode;
  className?: string;
}

export function ChartCard({ title, loading, children, className }: CardProps) {
  return (
    <div className={cn("rounded-xl border border-gray-200 bg-white p-4 shadow-sm", className)}>
      {title && <h3 className="mb-3 text-sm font-semibold text-gray-600 uppercase tracking-wide">{title}</h3>}
      {loading ? <Skeleton className="h-64 w-full" /> : children}
    </div>
  );
}

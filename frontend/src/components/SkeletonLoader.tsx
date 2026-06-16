export default function SkeletonLoader() {
  return (
    <div className="w-full space-y-4 animate-pulse" aria-busy="true" aria-label="Loading prediction…">

      {/* ResultCard skeleton */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-md space-y-4">
        <div className="flex items-center justify-between">
          <div className="h-4 w-28 rounded-full bg-white/10" />
          <div className="h-6 w-20 rounded-full bg-white/10" />
        </div>
        <div className="h-8 w-48 rounded-lg bg-white/10" />
        <div className="h-3 w-full rounded-full bg-white/10" />
        <div className="h-3 w-4/5 rounded-full bg-white/10" />
        <div className="grid grid-cols-3 gap-3 pt-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 rounded-xl bg-white/10" />
          ))}
        </div>
      </div>

      {/* ResumeDNA skeleton */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-md space-y-4">
        <div className="h-4 w-36 rounded-full bg-white/10" />
        {/* Radar chart placeholder */}
        <div className="mx-auto h-48 w-48 rounded-full bg-white/10" />
        {/* Skill pills */}
        <div className="flex flex-wrap gap-2 pt-2">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="h-6 w-20 rounded-full bg-white/10" />
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-6 w-24 rounded-full bg-white/10" />
          ))}
        </div>
      </div>

    </div>
  );
}

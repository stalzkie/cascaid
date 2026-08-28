export function scaleLinear([d0, d1]: [number, number], [r0, r1]: [number, number]): (v: number) => number {
  const domainSpan = d1 - d0;
  if (domainSpan === 0) {
    const mid = (r0 + r1) / 2;
    return () => mid;
  }
  return (v: number) => r0 + ((v - d0) / domainSpan) * (r1 - r0);
}

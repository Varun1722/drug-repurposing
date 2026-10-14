// The limit on expansion rounds for the web app, defined once.
//
// "Expansion rounds" are the times the reasoning step may widen the search after
// the first docking round. Docking round 1 is the initial one, so with a limit
// of N the last docking round is N + 1, and reasoning is never asked about a
// round >= N + 1.
//
// Used by app/api/analyze/route.ts (the prompt and the cutoff) and app/page.tsx
// (whether to offer an expansion), so the two cannot drift apart. It is a single
// fixed value; the old Python CLI took it as --max-rounds (see
// legacy-python-pipeline/).

export const MAX_EXPANSION_ROUNDS = 2

/** The docking round at which no further expansion is allowed. */
export const MAX_DOCKING_ROUND = MAX_EXPANSION_ROUNDS + 1

/** True if the model may be asked to expand after docking round `round`. */
export function mayExpandAfter(round: number): boolean {
  return round < MAX_DOCKING_ROUND
}

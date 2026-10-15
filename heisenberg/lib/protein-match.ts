// Which drugs are docked against which protein.
//
// A drug goes to a protein when the literature review listed that protein for it,
// compared as whole names (trimmed, case and spacing ignored). It was text inside
// the name before, both ways, so "ER" matched "HER2", "ACE" matched "ACE2" and "RAS"
// matched "KRAS", and a protein with no match was docked against every drug. In four
// real runs the second was the bigger harm: 67 to 93% of the docking pairs came from
// that fallback. The model copies the protein list exactly (97 of 97 listed names), so
// whole names lose nothing. A protein nothing is listed for, and a drug listed for none
// of the proteins, are returned so the page can say they were not docked.
// shortcut: no alias or family resolution (HER2 is ERBB2, PI3K is a family): a drug
// listed under another name for the same protein is not linked. Resolving names to
// genes was tried on the recorded runs and is unsafe ("Spike" resolves to the human gene
// CHMP5, "PI3K" to PIK3CA alone), so upgrade only with a reviewed alias list.

import type { ReviewDrug } from "./types"

const nameKey = (name: string) => name.trim().toLowerCase().replace(/\s+/g, " ")

export interface DrugLinks {
  /** `drugsByProtein[i]` are the drugs to dock against `proteins[i]`. */
  drugsByProtein: ReviewDrug[][]
  proteinsWithoutDrugs: string[]
  /** Names of the drugs the review listed for none of the proteins. */
  drugsWithoutProtein: string[]
}

export function linkDrugsToProteins(proteins: string[], drugs: ReviewDrug[]): DrugLinks {
  const drugsByProtein = proteins.map((protein) => {
    const key = nameKey(protein)
    return drugs.filter((d) => (d.proteins ?? []).some((listed) => nameKey(listed) === key))
  })
  const linked = new Set(drugsByProtein.flat())
  return {
    drugsByProtein,
    proteinsWithoutDrugs: proteins.filter((_, i) => drugsByProtein[i].length === 0),
    drugsWithoutProtein: drugs.filter((d) => !linked.has(d)).map((d) => d.drug),
  }
}

const some = (names: string[], max: number) =>
  names.length <= max ? names.join(", ") : `${names.slice(0, max).join(", ")} and ${names.length - max} more`

/** The chat note for what was left out of docking, or "" when nothing was. */
export function describeNotDocked({
  proteinsWithoutDrugs,
  drugsWithoutProtein,
}: Pick<DrugLinks, "proteinsWithoutDrugs" | "drugsWithoutProtein">): string {
  const notes: string[] = []
  if (proteinsWithoutDrugs.length > 0) {
    const many = proteinsWithoutDrugs.length > 1
    notes.push(`No drug was found for ${some(proteinsWithoutDrugs, 4)}, so ${many ? "they were" : "it was"} not docked.`)
  }
  if (drugsWithoutProtein.length > 0) {
    const many = drugsWithoutProtein.length > 1
    notes.push(
      `${drugsWithoutProtein.length} drug${many ? "s" : ""} from the review ${many ? "were" : "was"} not linked to any of the protein targets ` +
        `(${some(drugsWithoutProtein, 3)}), so ${many ? "they" : "it"} will not be docked.`
    )
  }
  return notes.join("\n\n")
}

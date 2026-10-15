// Which drugs are docked against which protein (lib/protein-match.ts).
//
// Contract: a drug goes to a protein when the review listed that protein for it,
// compared as whole names (trimmed, case and spacing ignored), never as text inside
// another name: "ER" is not in "HER2", "ACE" not in "ACE2", "RAS" not in "KRAS". There
// is no "dock every drug" fallback: a protein nothing is listed for is not docked, and
// a drug listed for none of the proteins is not docked; both are reported so the page
// can say so. The real-run cases come from four recorded /api/review runs
// (tests/fixtures/protein-names): the model copies the protein list exactly (97 of 97
// names), so exact names lose nothing; the old substring rule added 1 wrong pair in
// them, and its fallback made 67 to 93% of the docking pairs unrelated to any listing.

import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import { describeNotDocked, linkDrugsToProteins } from "../lib/protein-match"
import type { ReviewDrug } from "../lib/types"

const drug = (name: string, proteins: string[] | undefined): ReviewDrug =>
  ({ drug: name, proteins, mechanism: "m", fdaStatus: "s", category: "mainstream" }) as ReviewDrug

const forProtein = (proteins: string[], drugs: ReviewDrug[], protein: string) =>
  linkDrugsToProteins(proteins, drugs).drugsByProtein[proteins.indexOf(protein)].map((d) => d.drug)

describe("whole names, not text inside a name", () => {
  const cases: [string, string][] = [
    ["ER", "HER2"],
    ["ER", "ERBB2"],
    ["ACE", "ACE2"],
    ["RAS", "KRAS"],
    ["MET", "METAP2"],
    ["AR", "PARP1"],
    ["KRAS", "KRAS G12C"], // listed separately by the model, so kept separate
  ]
  for (const [protein, listed] of cases) {
    it(`a drug listed for "${listed}" is not docked against "${protein}"`, () => {
      assert.deepEqual(forProtein([protein, listed], [drug("d", [listed])], protein), [])
      assert.deepEqual(forProtein([protein, listed], [drug("d", [listed])], listed), ["d"])
    })
  }

  it("ignores case, spacing and surrounding whitespace", () => {
    assert.deepEqual(forProtein(["KRAS G12C"], [drug("d", ["  kras   g12c "])], "KRAS G12C"), ["d"])
    assert.deepEqual(forProtein(["PD-L1"], [drug("d", ["pd-l1"])], "PD-L1"), ["d"])
  })

  it("a drug listed for several proteins goes to each of them, and to no other", () => {
    const proteins = ["CDK4", "CDK6", "HER2"]
    const drugs = [drug("palbociclib", ["CDK4", "CDK6"])]
    assert.deepEqual(forProtein(proteins, drugs, "CDK4"), ["palbociclib"])
    assert.deepEqual(forProtein(proteins, drugs, "CDK6"), ["palbociclib"])
    assert.deepEqual(forProtein(proteins, drugs, "HER2"), [])
  })
})

describe("no fallback", () => {
  it("a protein nothing is listed for gets no drugs and is reported", () => {
    const links = linkDrugsToProteins(["EGFR", "BRCA2"], [drug("gefitinib", ["EGFR"])])
    assert.deepEqual(links.drugsByProtein.map((d) => d.map((x) => x.drug)), [["gefitinib"], []])
    assert.deepEqual(links.proteinsWithoutDrugs, ["BRCA2"])
  })

  it("a drug listed for none of the proteins is reported and docked against nothing", () => {
    const links = linkDrugsToProteins(
      ["EGFR"],
      [drug("metformin", []), drug("lithium", undefined), drug("odd", ["not in the list"]), drug("gefitinib", ["EGFR"])]
    )
    assert.deepEqual(links.drugsByProtein[0].map((d) => d.drug), ["gefitinib"])
    assert.deepEqual(links.drugsWithoutProtein, ["metformin", "lithium", "odd"])
  })

  it("no drugs at all: every protein is reported, nothing is docked", () => {
    const links = linkDrugsToProteins(["EGFR", "KRAS"], [])
    assert.deepEqual(links.proteinsWithoutDrugs, ["EGFR", "KRAS"])
    assert.deepEqual(links.drugsWithoutProtein, [])
  })
})

describe("on the four real runs", () => {
  const load = (file: string) =>
    JSON.parse(readFileSync(new URL(`./fixtures/protein-names/review-${file}.json`, import.meta.url), "utf8")) as {
      proteins: string[]
      drugs: ReviewDrug[]
    }
  const run = (file: string) => {
    const { proteins, drugs } = load(file)
    const links = linkDrugsToProteins(proteins, drugs)
    return { proteins, drugs, links, pairs: links.drugsByProtein.reduce((n, d) => n + d.length, 0) }
  }

  it("breast cancer: Cyclin D and BRCA2 have no drug (the fallback gave each of them all 151)", () => {
    const { links, pairs } = run("breast-cancer")
    assert.equal(pairs, 147)
    assert.deepEqual(links.proteinsWithoutDrugs, ["Cyclin D", "BRCA2"])
    assert.equal(links.drugsWithoutProtein.length, 8)
    assert.ok(links.drugsWithoutProtein.includes("Metformin"))
  })

  it("Alzheimer's: sTREM2 does not get the drug listed for TREM2 (the substring rule gave it)", () => {
    const { proteins, links, pairs } = run("alzheimer-s-disease")
    assert.equal(pairs, 70)
    assert.deepEqual(links.drugsByProtein[proteins.indexOf("TREM2")].map((d) => d.drug), ["AL002"])
    assert.deepEqual(links.drugsByProtein[proteins.indexOf("sTREM2")], [])
    assert.equal(links.proteinsWithoutDrugs.length, 10)
    assert.equal(links.drugsWithoutProtein.length, 18)
  })

  it("COVID-19: 13 of 16 proteins have no drug (the fallback made 585 of 630 pairs unrelated to any listing)", () => {
    const { proteins, links, pairs } = run("covid-19")
    assert.equal(pairs, 45)
    assert.equal(links.proteinsWithoutDrugs.length, 13)
    assert.equal(proteins.length - links.proteinsWithoutDrugs.length, 3)
  })

  it("NSCLC (no drugs came back): nothing is docked and every protein is reported", () => {
    const { proteins, links, pairs } = run("non-small-cell-lung-cancer")
    assert.equal(pairs, 0)
    assert.equal(links.proteinsWithoutDrugs.length, proteins.length)
  })

  it("every pair is a name the review listed for that drug", () => {
    for (const file of ["breast-cancer", "alzheimer-s-disease", "covid-19"]) {
      const { proteins, links } = run(file)
      proteins.forEach((protein, i) => {
        for (const d of links.drugsByProtein[i]) {
          assert.ok(d.proteins.some((p) => p.trim().toLowerCase() === protein.toLowerCase()), `${file}: ${d.drug} for ${protein}`)
        }
      })
    }
  })
})

describe("describeNotDocked (the chat note)", () => {
  const note = (proteinsWithoutDrugs: string[], drugsWithoutProtein: string[]) =>
    describeNotDocked({ proteinsWithoutDrugs, drugsWithoutProtein })

  it("says nothing when everything is linked", () => {
    assert.equal(note([], []), "")
  })

  it("names the proteins that have no drug, singular and plural", () => {
    assert.equal(note(["BRCA2"], []), "No drug was found for BRCA2, so it was not docked.")
    assert.equal(note(["Cyclin D", "BRCA2"], []), "No drug was found for Cyclin D, BRCA2, so they were not docked.")
  })

  it("names the drugs the review listed for none of the proteins, and does not call that a missing structure", () => {
    assert.equal(
      note([], ["Metformin"]),
      "1 drug from the review was not linked to any of the protein targets (Metformin), so it will not be docked."
    )
    assert.match(note([], ["a", "b"]), /^2 drugs from the review were not linked to any of the protein targets \(a, b\), so they will not be docked\.$/)
  })

  it("shows a few names and counts the rest", () => {
    assert.match(note(["a", "b", "c", "d", "e", "f"], []), /^No drug was found for a, b, c, d and 2 more, so they were not docked\.$/)
    assert.match(note([], ["a", "b", "c", "d", "e"]), /\(a, b, c and 2 more\)/)
  })

  it("puts both notes in one paragraph break apart", () => {
    assert.equal(note(["X"], ["Y"]).split("\n\n").length, 2)
  })
})

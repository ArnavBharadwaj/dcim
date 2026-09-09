/**
 * Combined report: the air-cooled twin and the liquid-cooled twin in one document.
 *
 * Times New Roman throughout, black on white, colour only inside the figures --
 * the same treatment as the two PDFs.
 *
 *   node scripts/make_combined_docx.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, TableOfContents, convertInchesToTwip,
} = require("docx");

const REPO = path.resolve(__dirname, "..");
const AIR = path.join(REPO, "results/figures/report");
const LIQ = path.join(REPO, "results/figures/liquid_report");

const CONTENT_DXA = 9638;            // A4 minus 2 cm margins each side
const BLACK = "000000";
const GREY = "555555";
const RULE = "000000";
const FAINT = "BBBBBB";
const HEADFILL = "EFEFEF";

/* ------------------------------------------------------------------ helpers */

const p = (text, opts = {}) =>
  new Paragraph({
    alignment: opts.align ?? AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 120, line: opts.line ?? 264 },
    indent: opts.indent,
    children: runs(text, opts),
  });

// Inline markup: **bold**, _italic_. Kept deliberately small.
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|_[^_]+_)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(mk(text.slice(last, m.index), opts));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(mk(tok.slice(2, -2), { ...opts, bold: true }));
    else out.push(mk(tok.slice(1, -1), { ...opts, italics: true }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(mk(text.slice(last), opts));
  return out;
}
const mk = (t, o = {}) =>
  new TextRun({
    text: t, bold: o.bold, italics: o.italics,
    size: o.size ?? 20, color: o.color ?? BLACK,
    font: o.mono ? "Consolas" : undefined,
  });

const h1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 130 },
  children: [new TextRun({ text: t, bold: true, size: 26, color: BLACK })],
});
const h2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 90 },
  children: [new TextRun({ text: t, bold: true, size: 22, color: BLACK })],
});
const partHead = (t, sub) => [
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 200, after: 60 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: RULE, space: 6 } },
    children: [new TextRun({ text: t, bold: true, size: 30, color: BLACK })],
  }),
  new Paragraph({
    spacing: { after: 200 },
    children: [new TextRun({ text: sub, italics: true, size: 21, color: GREY })],
  }),
];
const note = (t) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 140, line: 240 },
  children: runs(t, { size: 17, color: GREY }),
});
const hr = () => new Paragraph({
  spacing: { before: 40, after: 140 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: FAINT, space: 2 } },
  children: [],
});

function figure(dir, file, widthPx, natural, caption) {
  const [w, h] = natural;
  const img = new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 140, after: 60 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path.join(dir, file)),
      transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
    })],
  });
  const cap = new Paragraph({
    alignment: AlignmentType.LEFT, spacing: { after: 200, line: 230 },
    children: runs(caption, { size: 17, color: GREY }),
  });
  return [img, cap];
}

function table(rows, widths, rightCols = []) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map((x) => Math.round(x / total * CONTENT_DXA));
  const body = rows.map((cells, ri) =>
    new TableRow({
      tableHeader: ri === 0,
      children: cells.map((c, ci) => new TableCell({
        width: { size: scaled[ci], type: WidthType.DXA },
        shading: ri === 0
          ? { type: ShadingType.CLEAR, fill: HEADFILL, color: "auto" }
          : undefined,
        margins: { top: 60, bottom: 60, left: 90, right: 90 },
        children: [new Paragraph({
          alignment: rightCols.includes(ci) ? AlignmentType.RIGHT : AlignmentType.LEFT,
          spacing: { after: 0, line: 230 },
          children: runs(String(c), { size: 17, bold: ri === 0 }),
        })],
      })),
    }));
  return new Table({
    columnWidths: scaled,
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 8, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 8, color: RULE },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: FAINT },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    rows: body,
  });
}

const spacer = (after = 160) => new Paragraph({ spacing: { after }, children: [] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

/* --------------------------------------------------------------- document */

const kids = [];

kids.push(new Paragraph({
  spacing: { after: 60 },
  children: [new TextRun({
    text: "Digital Twin-Assisted Workload Allocation and Cooling Optimisation",
    bold: true, size: 38, color: BLACK })],
}));
kids.push(new Paragraph({
  spacing: { after: 60 },
  children: [new TextRun({
    text: "Two thermal twins: air-cooled recirculation and direct-to-chip liquid",
    italics: true, size: 22, color: BLACK })],
}));
kids.push(new Paragraph({
  spacing: { after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 14, color: RULE, space: 6 } },
  children: [],
}));

kids.push(p("This document combines the two thermal models built for this project and the results measured from each. It is written to be checked rather than trusted: every number below is produced by a script in the repository and logged to _results/runs.csv_ with its configuration hash, seed and git commit.", { size: 21, after: 140 }));

kids.push(table([
  ["Phases complete", "Tests", "Simulated time", "Logged runs", "Halls"],
  ["0, 0b, 1, 2, 3", "167 passing", "1,500 hours", "173 rows", "3 air, 1 liquid"],
], [24, 16, 21, 17, 22]));
kids.push(spacer(200));

kids.push(h1("Summary"));
kids.push(p("The project set out to build a digital twin of a data hall whose internal state is a directed graph, train a message-passing network to predict per-rack temperature, and drive two controllers from that forecast."));
kids.push(p("Two measurements reshaped the plan. The first was that the chosen simulator has no thermal physics at all: its rack inlet temperature is a constant plus the cooling setpoint, so a 14.8 kW swing on one rack moves every rack by exactly zero. That forced the heat model to be rewritten before any comparison could mean anything."));
kids.push(p("The second was subtler and more consequential. On the resulting air-cooled twin, neighbouring racks' temperatures correlate at **0.9953**, so spatial context adds very little to single-rack accuracy. The graph model still wins at every horizon and transfers to unseen halls, but the margin is modest and the edge-shuffle ablation should be expected to come back near-null."));
kids.push(p("That result is a property of **air**, not of data halls. Air recirculation is diffuse, so every rack in a region shares one thermal environment. A direct-to-chip liquid-cooled hall is plumbed rather than ventilated: coupling follows the coolant topology, and racks standing side by side on different loops are thermally independent. Part II builds that hall and measures the difference."));
kids.push(p("**One caveat governs everything below.** After the Phase 0 rewrite, both thermal models are ours. The only measured data anywhere in this project is the Alibaba GPU workload trace. Part III sets out exactly what is real, what is plausible, and what was invented."));

kids.push(pageBreak());

/* --------------------------------------------------- Part I: air */
kids.push(...partHead("Part I  ·  The air-cooled twin",
  "Heat recirculation, and the ceiling it puts on what a graph model can add"));

kids.push(h1("1.  The Phase 0 finding"));
kids.push(p("SustainDC computes rack inlet temperature in exactly one place, _envs/datacenter.py:279_:"));
kids.push(new Paragraph({
  spacing: { before: 60, after: 120 },
  indent: { left: convertInchesToTwip(0.3) },
  children: [new TextRun({
    text: "rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint",
    font: "Consolas", size: 19, color: BLACK })],
}));
kids.push(p("The approach temperature is a constant read from JSON at startup and never recomputed. No rack's power enters — not a neighbour's, not the rack's own. Driving one rack from 50% to 100% utilisation, a 14.8 kW change, moves every rack's inlet temperature by exactly 0.000e+00 K against a float64 spacing of 3.6e−15."));
kids.push(p("Every model in the planned comparison would therefore have tied at machine precision, and every ablation would have been null by construction. Two further blockers surfaced while probing: the gym wrapper hardcodes one scalar utilisation for all racks, so placement was inexpressible, and three of the four shipped hall configurations do not load."));
kids.push(...figure(AIR, "fig1_coupling.png", 610, [1620, 870],
  "**Figure 1.** Response of every rack's inlet temperature to a 14.8 kW perturbation at one rack. SustainDC's values are exactly zero at every distance, including zero distance; its curve stops at distance 7 because its default hall holds only 20 racks. The replacement kernel decays smoothly across the hall."));

kids.push(h1("2.  The replacement model"));
kids.push(p("With a_ij the fraction of rack i's exhaust drawn into rack j's inlet and K the diagonal matrix of rack thermal mass flows, the mixing energy balance at each inlet rearranges to T = T_supply + D·P, where D = K⁻¹[(I − Aᵀ)⁻¹ − I]. The matrix inverse expands as a Neumann series in Aᵀ, so multi-hop coupling emerges even though A is local: term m is air that passed through m racks before arriving."));
kids.push(p("A is built from the floor plan alone — distance, rows crossed, and direction relative to the return-air path — never from the kernel the model is being asked to learn. The coupling is state-dependent: K moves with rack fan speed and leakage rises when racks draw more air than the CRACs supply, so a cooling action reshapes the coupling rather than merely shifting it. SustainDC's setpoint gain was exactly 1.000000 at every rack; the replacement's ranges from 0.878 to 0.923."));
kids.push(h2("A modelling bug worth recording"));
kids.push(p("Normalising each source's leakage by geometry alone leaves a rack's recirculated intake fixed while its total intake grows with fan speed. Ramping its fans then dilutes its own inlet, and loading a rack from 50% to 100% utilisation _cooled_ it by 0.79 K while its neighbours warmed by 0.08 K. A placement controller trained against that would have learned to stack load onto the hottest racks. Weighting each target's share of the recirculating air by its own thermal mass flow removes the artefact; two regression tests hold it removed."));

kids.push(h1("3.  The dataset"));
kids.push(p("Real Alibaba PAI GPU jobs are replayed onto racks under four placement policies, including a deliberately bad corner-stacking policy so the dataset contains hot states. Cooling setpoints are held for 5 to 20 minutes and then stepped, as a real CRAC controller behaves. 500 hours per hall at 30-second resolution across three halls, split by episode so that no input window crosses a boundary and every training timestep precedes every test timestep."));
kids.push(note("Three calibration errors were caught by measuring the output rather than trusting the config: a cooling schedule that moved the supply temperature only 0.68 K across a five-hour episode, collapsing the 30 s prediction target to 0.03 K; a supply cap that left less headroom than the hall's own recirculation rise, putting 67% of timesteps in violation; and fan speeds low enough to push 4.7% past the allowable limit."));

kids.push(h1("4.  Results"));
kids.push(h2("4.1  Accuracy on hall_a"));
kids.push(p("Test RMSE of the temperature delta, in kelvin, over five seeds."));
kids.push(table([
  ["Model", "Spatial input", "30 s", "60 s", "300 s"],
  ["Persistence", "none", "0.1057", "0.1908", "0.6367"],
  ["RC network", "graph edges", "0.1682", "0.1978", "0.4833"],
  ["Per-rack LSTM", "none", "0.0272", "0.0598", "0.2245"],
  ["LightGBM k=4", "4 nearest racks", "0.0433", "0.0606", "0.1884"],
  ["**Graph model", "**message passing", "**0.0244", "**0.0434", "**0.1677"],
], [22, 26, 17, 17, 18], [2, 3, 4]));
kids.push(spacer(80));
kids.push(note("Margin over the best baseline: 11% at 30 s, 27% at 60 s, 11% at 300 s. Seed spread is ±0.003 to ±0.004, so the ordering is stable."));

kids.push(h2("4.2  Why the margin is modest"));
kids.push(p("LightGBM with the 2, 4 and 6 nearest racks scores identically to three decimal places at every horizon. That is not a bug. Own-rack features alone reach linear R² = 0.8251 on the 300 s target; adding every neighbour feature moves it to 0.8265; neighbour features alone reach 0.0280. Neighbouring racks share almost exactly the same recirculation environment, so a rack's own temperature already encodes what its neighbours are experiencing."));
kids.push(...figure(AIR, "fig3_redundancy.png", 300, [780, 750],
  "**Figure 2.** A rack's own inlet temperature against the mean of its six nearest racks, hall_a. The dashed line is equality; the scatter collapses onto it at r = 0.9952."));

kids.push(h2("4.3  Zero-shot transfer"));
kids.push(p("The graph model is trained on hall_a and applied unchanged to two other halls, carrying hall_a's normalisation statistics with it. Those statistics are never recomputed on a target hall, which would be leakage. Results are given as skill — the fraction of the trivial predictor's error removed — because the halls have different target variance and raw RMSE is not comparable across them."));
kids.push(table([
  ["Hall", "Relationship to hall_a", "30 s", "60 s", "300 s"],
  ["hall_a", "in-domain, 200 racks", "76.0%", "75.0%", "70.8%"],
  ["hall_b", "140 racks, same aisle pattern", "**81.6%", "**78.9%", "**71.7%"],
  ["hall_c", "uncontained layout, moved CRACs", "71.6%", "74.4%", "68.2%"],
], [13, 37, 16, 16, 18], [2, 3, 4]));
kids.push(spacer(80));
kids.push(p("Transfer holds. On hall_b the transferred model removes more of persistence's error than it does in-domain, because that hall's larger target variance makes the trivial predictor worse. Against baselines **refit on the target hall**, the zero-shot graph model still wins on hall_b (0.0263 against the LSTM's 0.0396 at 30 s) but loses on hall_c (0.0292 against 0.0224). Scale transfer beats local retraining; layout transfer does not."));
kids.push(...figure(AIR, "fig6_transfer.png", 610, [1620, 780],
  "**Figure 3.** Skill against each hall's own persistence baseline. The transferred model removes as much of the trivial predictor's error on halls it has never seen as it does in-domain."));

kids.push(pageBreak());

/* --------------------------------------------------- Part II: liquid */
kids.push(...partHead("Part II  ·  The liquid-cooled twin",
  "Direct-to-chip cooling, where coupling follows the plumbing"));

kids.push(h1("5.  Why liquid cooling changes the question"));
kids.push(p("Air recirculation is diffuse. Every rack in a region shares one thermal environment, so neighbour features are close to redundant — that is the 0.9953 correlation of Section 4.2, and it is why the edge-shuffle ablation on the air twin was expected to come back near-null."));
kids.push(p("A direct-to-chip hall is plumbed, not ventilated. Coolant leaves a Coolant Distribution Unit, runs along a branch manifold, and is drawn off in parallel by the racks on that branch. Racks sharing a branch share a supply temperature, a flow budget and a CDU. Racks on a different loop share almost nothing — however close they stand."));
kids.push(...figure(LIQ, "L1_topology.png", 610, [1680, 870],
  "**Figure 4.** hall_l1: 128 racks, four CDUs, two branch manifolds each. Colour is coolant loop, not position. Rows that sit next to each other on the floor belong to different loops, and the marked pair stands 2.4 m apart with no hydraulic connection at all."));

kids.push(h1("6.  The model"));
kids.push(p("Cold plates take 80% of rack power into coolant; the remaining fifth leaves as air and recirculates through the air twin's kernel unchanged, with rack airflow scaled to the air share because a liquid-cooled rack ships far smaller fans. At each CDU the secondary supply sits above the facility water by an approach that degrades as the unit approaches its rated duty, which is what couples every rack on a loop to every other rack on it. Along a branch, flow droops with distance from the feed and the supply is warmed by the return of every rack upstream."));
kids.push(h2("Two errors found by measuring rather than assuming"));
kids.push(p("**Cross-talk is what makes the graph non-trivial.** The first version had only a static manifold gain, so perturbing any rack moved every rack on its loop by exactly the same amount and loop topology explained **100%** of the coupling variance. That is not a graph problem: one categorical “which CDU” feature would capture the whole structure. Adding supply-return manifold cross-talk made the coupling directed and ordered within a branch, and explained variance fell to 0.645."));
kids.push(p("**The case-to-coolant rise is per device, not per rack.** It was first written as kelvin per kilowatt of _rack_ power, which made a denser rack appear to run each accelerator hotter. Every accelerator has its own cold plate fed in parallel off the rack manifold, so the rise is set by per-device power."));

kids.push(h1("7.  What the coupling probe measures"));
kids.push(p("Each rack is given a 5 kW step and the coolant supply response is measured at every other rack. hall_l1 at 85% utilisation, 34 °C facility water, pumps at 85%."));
kids.push(table([
  ["Relationship", "Pairs", "Mean |ΔT| (K)", "Mean gap (m)"],
  ["same branch (shared manifold)", "1,920", "**0.01817", "3.40"],
  ["same CDU, different branch", "2,048", "0.00960", "4.24"],
  ["different CDU", "12,288", "**0.00000", "8.99"],
], [40, 16, 24, 20], [1, 2, 3]));
kids.push(spacer(100));
kids.push(p("The most strongly coupled group is not the physically closest. Restricting to racks within one row pitch of each other makes the point sharper: of 1,508 such pairs, the 1,232 on the same CDU couple at 0.01753 K and the **276 on different CDUs couple at exactly zero**. Those racks stand side by side and are thermally independent. A k=4 neighbour feature set built on Euclidean distance would select them and learn from channels carrying no signal."));
kids.push(p("Over all pairs, coupling correlates with distance at −0.448, while loop topology explains 0.645 of its variance."));
kids.push(...figure(LIQ, "L2_coupling.png", 610, [1680, 780],
  "**Figure 5.** Left: coupling against physical distance, coloured by hydraulic relationship. The three bands are flat — distance carries almost no information about coupling. Zeros are drawn on the axis floor because the scale is logarithmic. Right: the same values grouped by relationship."));
kids.push(...figure(LIQ, "L3_matrix.png", 400, [1050, 900],
  "**Figure 6.** The influence matrix with racks ordered by loop, then by manifold position. Four blocks, nothing between them. The triangular shading inside each block is the cross-talk term: upstream racks warm downstream ones and not the reverse. This figure also makes the circularity discussed in Section 9 plain — the black off-diagonal blocks are imposed, not measured."));

kids.push(h1("8.  Operating envelope"));
kids.push(p("Across the ASHRAE liquid classes W32 to W45. At the cold end the hall has easy margin; at the warm end, where a hall with no mechanical chilling actually runs, the 85 °C case limit binds and pump speed is what saves it."));
kids.push(table([
  ["Load", "Facility water", "Pumps", "Coolant supply", "Peak case", "Throttling"],
  ["90%", "26 °C", "100%", "33.2", "60.4", "no"],
  ["90%", "34 °C", "100%", "41.3", "69.5", "no"],
  ["90%", "42 °C", "100%", "49.5", "78.6", "no"],
  ["100%", "42 °C", "80%", "50.0", "84.6", "no"],
  ["100%", "42 °C", "55%", "50.1", "**91.1", "**yes"],
  ["60%", "42 °C", "55%", "48.4", "77.6", "no"],
], [11, 21, 13, 21, 17, 17], [3, 4]));
kids.push(spacer(100));
kids.push(p("This gives the cooling controller two levers with different costs, which the air hall did not have. Facility water temperature is slow and plant-wide. Pump speed is fast, per-loop, and moves chip temperature through the cold plate's convective coefficient without moving the coolant supply at the head of a branch at all."));
kids.push(...figure(LIQ, "L4_envelope.png", 610, [1680, 810],
  "**Figure 7.** Peak chip case temperature at full load across the warm-water band, by pump speed. The curves are nearly parallel, so a hall can trade pumping energy against how warm its facility water is allowed to run."));

kids.push(pageBreak());

/* --------------------------------------------------- Part III */
kids.push(...partHead("Part III  ·  Provenance and open items",
  "What is real, what is not, and what has to happen before any of this is a claim"));

kids.push(h1("9.  What is and is not real"));
kids.push(p("**No part of either thermal model comes from a real data centre.** The only measured data anywhere in this project is the Alibaba PAI GPU workload trace: 100,000 real jobs over 289.8 hours. Every thermal number in Parts I and II is ours."));
kids.push(p("The liquid model carries a sharper version of this problem. Its headline result — that racks on different coolant loops show exactly zero coupling — is not a discovery. It is a consequence of how the model is written: the CDU approach temperature is a function of that CDU's own duty, so racks on different loops cannot influence each other, and the probe then measures that they do not. A reviewer will see this immediately."));
kids.push(p("The air twin is less circular in this respect. There, D = K⁻¹[(I − Aᵀ)⁻¹ − I] produced multi-hop coupling that was never specified: the 70 / 21 / 6 % hop profile emerged from a matrix inverse over a local kernel. The liquid model has much less of that. Its one genuinely emergent part is the ordering within a branch produced by manifold cross-talk, which is why loop topology explains 0.645 of the coupling variance rather than 1.000."));
kids.push(table([
  ["Tier", "What", "Status"],
  ["Measured", "Alibaba PAI GPU workload trace, 100,000 jobs over 289.8 h", "real"],
  ["Structural", "CDU / branch manifold / cold plate architecture; the roughly 80:20 liquid-air heat split; ASHRAE W32–W45 liquid classes; ~77 kW racks and an 85 °C case limit; hot-aisle/cold-aisle geometry and ASHRAE A1 air limits", "real, uncited"],
  ["Plausible", "Air: recirculation decay length, escape fraction, 60 s time constant. Liquid: design ΔT 12 K, 90 L/min per rack, CDU approach 2–8 K. Inside published ranges, but not read off a datasheet and not fitted to anything", "unvalidated"],
  ["Invented", "manifold_crosstalk 0.16; flow_droop_per_m 0.006; cdu_approach_gain 3.0; case_flow_exponent 0.40; the air kernel's directional weights", "**no source"],
], [13, 63, 24]));
kids.push(spacer(100));
kids.push(note("In reality racks on different CDUs _are_ weakly coupled, through the shared facility water loop, the shared air volume and shared pumping plant. The air path is modelled; the facility-water return is not, so the reported separation is cleaner than a real hall's would be."));

kids.push(h1("10.  Open items"));
kids.push(table([
  ["#", "Action", "Why"],
  ["1", "Model the shared facility water loop", "Cross-CDU coupling becomes small but non-zero rather than stipulated zero. Cheapest fix, and it removes the worst of the circularity in Section 9."],
  ["2", "Sweep manifold_crosstalk", "If the graph model's advantage exists at 0.16 and vanishes at 0.05, the liquid result is an artefact of a number with no source."],
  ["3", "Re-run Phase 3 with memory headroom", "The graph model's headline number came from a network of hidden width 32 with two attention heads, trained on 210,000 rows against LightGBM's 400,000, forced by an 8.6 GB machine. The confound runs against the graph model, which won regardless, but it should not have to be explained away."],
  ["4", "Anchor the CDU curve to a datasheet", "Vendors publish approach-against-load curves. This converts an invented parameter into a cited one."],
  ["5", "Run Phase 6 before Phase 5", "The edge-shuffle ablation directly tests whether the graph is decorative. On the air twin it should come back near-null; on the liquid twin it should be devastating. That contrast is the paper's strongest available result, and it is cheap."],
  ["6", "State the framing precisely", "Not “evaluated on a liquid-cooled data hall” but “evaluated on a parametric direct-to-chip model whose coupling structure is imposed by construction”."],
], [5, 30, 65]));

kids.push(h1("11.  What remains unbuilt"));
kids.push(p("Phases 4, 5 and 6 have not been started: online calibration and drift injection, which is what distinguishes a twin from a surrogate; the placement and cooling controllers the title promises; and the five ablations. The liquid twin has a thermal model, a graph and a coupling probe, but no generated trajectories and no trained model — Parts I's Phases 1 through 3 have not been repeated for it."));

const doc = new Document({
  creator: "Arnav Bharadwaj",
  title: "Digital twin build report: air and liquid cooling",
  description: "Combined report on the air-cooled and liquid-cooled thermal twins",
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 20, color: BLACK } },
      heading1: { run: { font: "Times New Roman", color: BLACK } },
      heading2: { run: { font: "Times New Roman", color: BLACK } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 },
      },
    },
    children: kids,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(REPO, "results", "twin-report-air-and-liquid.docx");
  fs.writeFileSync(out, buf);
  console.log(`wrote ${path.relative(REPO, out)}  (${(buf.length / 1e6).toFixed(2)} MB)`);
});

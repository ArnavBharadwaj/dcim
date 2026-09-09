"""Build the liquid-cooling PDF report.

Same treatment as the build report: Times New Roman, black on white, colour only inside
the figures.

    uv run python scripts/make_liquid_pdf.py
"""
from __future__ import annotations

import pathlib
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.make_pdf_report import (BLACK, F, FAINT, FB, FI, GREY, RULE,  # noqa: E402
                                     body, cap, cell, cellb, figure, h1, h2,
                                     lead, rule, small, sub, table, title)

FIG = REPO / "results" / "figures" / "liquid_report"
OUT = REPO / "results" / "liquid-cooling-report.pdf"


def liquid_figure(name, caption, width=138 * mm):
    from scripts.make_pdf_report import figure_parts
    from reportlab.platypus import KeepTogether
    p = FIG / name
    from PIL import Image as PILImage
    from reportlab.platypus import Image
    with PILImage.open(p) as im:
        w, h = im.size
    return KeepTogether([Image(str(p), width=width, height=width * h / w),
                         Paragraph(caption, cap)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 7.6)
    canvas.setFillColor(GREY)
    canvas.drawString(20 * mm, 12 * mm,
                      "Liquid-cooled twin  ·  direct-to-chip  ·  commit 9667dab")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(FAINT)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
    canvas.restoreState()


def boxed(flowables, accent=BLACK):
    """A ruled box for the passages that must not be skimmed past."""
    t = Table([[flowables]], colWidths=[None], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2.0, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm,
                          title="Liquid-cooled twin: direct-to-chip AI hall",
                          author="Arnav Bharadwaj")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    W = doc.width
    st = []

    st += [
        Paragraph("A Direct-to-Chip Liquid-Cooled Twin for AI Data Halls", title),
        Paragraph("Coupling that follows the plumbing, not the floor plan", sub),
        rule(1.1, RULE, 4),
        Paragraph(
            "The air-cooled twin put a ceiling on what this paper could claim: "
            "neighbouring racks' inlet temperatures correlate at 0.9953, so spatial "
            "context adds almost nothing to single-rack accuracy and a graph model has "
            "little room to win. This report describes a liquid-cooled hall built to "
            "test whether that ceiling is a property of data halls or a property of air.",
            lead),
        rule(0.4, FAINT, 4),
    ]

    # ---------------- provenance warning, up front ----------------
    st += [Paragraph("1.  What is and is not real", h1)]
    st += [boxed([
        Paragraph(
            "<b>No part of this thermal model comes from a real data centre.</b> The only "
            "measured data anywhere in this project is the Alibaba PAI GPU workload "
            "trace. Every thermal number below is ours.", body),
        Paragraph(
            "More sharply: the headline result &mdash; that racks on different coolant "
            "loops show <i>exactly zero</i> coupling &mdash; is not a discovery. It is a "
            "consequence of how the model is written. The CDU approach temperature is a "
            "function of that CDU's own duty, so racks on different loops cannot "
            "influence each other, and the probe then measures that they do not. A "
            "reviewer will see this immediately, and the paper must not present it as an "
            "empirical finding.", body),
    ], BLACK)]
    st += [Spacer(1, 6)]
    st += [Paragraph(
        "The air twin was less circular in this respect. There, "
        "<font name='%s'>D = K<super>-1</super>[(I &minus; A<super>T</super>)"
        "<super>-1</super> &minus; I]</font> produced multi-hop coupling that was never "
        "specified: the 70 / 21 / 6 %% hop profile emerged from a matrix inverse over a "
        "local kernel. The liquid model has much less of that. The one genuinely "
        "emergent part is the ordering within a branch produced by manifold cross-talk, "
        "which is why loop topology explains 0.645 of the coupling variance rather than "
        "1.000. That residual is the only part of the structure doing real work." % FI,
        body)]

    prov = [
        ["**Tier", "**What", "**Status"],
        ["Measured", "Alibaba PAI GPU workload trace, 100,000 jobs over 289.8 h",
         "real"],
        ["Structural",
         "CDU / branch manifold / cold plate architecture; the roughly 80:20 liquid-air "
         "heat split; ASHRAE W32&ndash;W45 liquid classes; ~77 kW racks and an 85 &deg;C "
         "case limit, the right order for GB200-class hardware",
         "real, uncited"],
        ["Plausible",
         "Design &Delta;T 12 K; 90 L/min per rack; CDU approach 2&ndash;8 K. Inside "
         "published vendor envelopes, but not read off a datasheet and not fitted to "
         "anything",
         "unvalidated"],
        ["Invented",
         "manifold_crosstalk 0.16; flow_droop_per_m 0.006; cdu_approach_gain 3.0; "
         "case_flow_exponent 0.40. Chosen to produce behaviour of the right shape",
         "**no source"],
    ]
    st += [table(prov, [W * .14, W * .62, W * .24])]
    st += [Spacer(1, 3), Paragraph(
        "In reality racks on different CDUs <i>are</i> weakly coupled, through the "
        "shared facility water loop, the shared air volume and shared pumping plant. "
        "The air path is modelled; the facility-water return is not, so the reported "
        "separation is cleaner than a real hall's would be.", small)]

    # ---------------- why liquid ----------------
    st += [Paragraph("2.  Why liquid cooling changes the question", h1)]
    st += [Paragraph(
        "Air recirculation is diffuse. Every rack in a region shares one thermal "
        "environment, so a rack's own temperature already encodes what its neighbours "
        "are experiencing, and neighbour features are close to redundant. That is the "
        "0.9953 correlation, and it is why the edge-shuffle ablation on the air twin was "
        "expected to come back near-null.", body)]
    st += [Paragraph(
        "A direct-to-chip hall is plumbed, not ventilated. Coolant leaves a Coolant "
        "Distribution Unit, runs along a branch manifold, and is drawn off in parallel "
        "by the racks on that branch. Racks sharing a branch share a supply temperature, "
        "a flow budget and a CDU. Racks on a different loop share almost nothing &mdash; "
        "however close they stand.", body)]
    st += [liquid_figure("L1_topology.png",
        "<b>Figure 1.</b> hall_l1: 128 racks, four CDUs, two branch manifolds each. "
        "Colour is coolant loop, not position. Rows that sit next to each other on the "
        "floor belong to different loops, and the marked pair stands 2.4 m apart with no "
        "hydraulic connection at all.")]

    # ---------------- the model ----------------
    st += [Paragraph("3.  The model", h1)]
    st += [Paragraph(
        "Cold plates take 80% of rack power into coolant; the remaining fifth leaves as "
        "air and recirculates through the air twin's kernel unchanged, with rack airflow "
        "scaled to the air share because a liquid-cooled rack ships far smaller fans. At "
        "each CDU the secondary supply sits above the facility water by an approach that "
        "degrades as the unit approaches its rated duty, which is what couples every rack "
        "on a loop to every other rack on it. Along a branch, flow droops with distance "
        "from the feed and the supply is warmed by the return of every rack upstream.",
        body)]

    st += [Paragraph("Two errors found by measuring rather than assuming", h2)]
    st += [Paragraph(
        "<b>Cross-talk is what makes the graph non-trivial.</b> The first version had "
        "only a static manifold gain, so perturbing any rack moved every rack on its loop "
        "by exactly the same amount, and loop topology explained <b>100%</b> of the "
        "coupling variance. That is not a graph problem: one categorical &ldquo;which "
        "CDU&rdquo; feature would capture the whole structure. Adding supply-return "
        "manifold cross-talk made the coupling directed and ordered within a branch, and "
        "explained variance fell to 0.645.", body)]
    st += [Paragraph(
        "<b>The case-to-coolant rise is per device, not per rack.</b> It was first "
        "written as kelvin per kilowatt of <i>rack</i> power, which made a denser rack "
        "appear to run each accelerator hotter. Every accelerator has its own cold plate "
        "fed in parallel off the rack manifold, so the rise is set by per-device power.",
        body)]
    st += [liquid_figure("L5_branch.png",
        "<b>Figure 2.</b> Along one branch manifold at 90% utilisation, 34 &deg;C "
        "facility water, pumps at 85%. Supply warms with distance from the feed because "
        "the return manifold alongside it carries the heat of every upstream rack; flow "
        "falls over the same span. A rack at the far end is therefore penalised twice.")]

    # ---------------- the probe ----------------
    st += [Paragraph("4.  What the probe measures", h1)]
    st += [Paragraph(
        "Each rack is given a 5 kW step and the coolant supply response is measured at "
        "every other rack. hall_l1 at 85% utilisation, 34 &deg;C facility water, pumps "
        "at 85%.", body)]
    res = [
        ["**Relationship", "**Pairs", "**Mean |&Delta;T| (K)", "**Mean gap (m)"],
        ["same branch (shared manifold)", "1,920", "**0.01817", "3.40"],
        ["same CDU, different branch", "2,048", "0.00960", "4.24"],
        ["different CDU", "12,288", "**0.00000", "8.99"],
    ]
    st += [table(res, [W * .40, W * .16, W * .24, W * .20], align_right=(1, 2, 3))]
    st += [Spacer(1, 5), Paragraph(
        "The most strongly coupled group is not the physically closest. Restricting to "
        "racks within one row pitch of each other makes the point sharper:", body)]
    near = [
        ["**Physically within 2.6 m", "**Pairs", "**Mean |&Delta;T| (K)"],
        ["on the same CDU", "1,232", "0.01753"],
        ["on different CDUs", "**276", "**0.00000"],
    ]
    st += [table(near, [W * .48, W * .22, W * .30], align_right=(1, 2))]
    st += [Spacer(1, 5), Paragraph(
        "<b>276 pairs of racks stand side by side and are thermally independent.</b> A "
        "<i>k</i>=4 neighbour feature set built on Euclidean distance would select those "
        "racks and learn from channels carrying no signal. Over all pairs, coupling "
        "correlates with distance at &minus;0.448, while loop topology explains 0.645 of "
        "its variance.", body)]
    st += [liquid_figure("L2_coupling.png",
        "<b>Figure 3.</b> Left: coupling against physical distance, coloured by "
        "hydraulic relationship. The three bands are flat &mdash; distance carries almost "
        "no information about coupling. Zeros are drawn on the axis floor because the "
        "scale is logarithmic. Right: the same values grouped by relationship.")]
    st += [liquid_figure("L3_matrix.png",
        "<b>Figure 4.</b> The influence matrix with racks ordered by loop and then by "
        "manifold position. Four blocks, nothing between them. The triangular shading "
        "inside each block is the cross-talk term: upstream racks warm downstream ones "
        "and not the reverse. This figure also makes the circularity of Section 1 "
        "plain &mdash; the black off-diagonal blocks are imposed, not measured.",
        104 * mm)]

    # ---------------- envelope ----------------
    st += [Paragraph("5.  Operating envelope", h1)]
    st += [Paragraph(
        "Across the ASHRAE liquid classes W32 to W45. At the cold end the hall has easy "
        "margin; at the warm end, where a hall with no mechanical chilling actually runs, "
        "the 85 &deg;C case limit binds and pump speed is what saves it. This gives the "
        "cooling controller two levers with different costs, which the air hall did not "
        "have: facility water temperature is slow and plant-wide, while pump speed is "
        "fast, per-loop, and moves chip temperature through the cold plate's convective "
        "coefficient without moving the coolant supply at the head of a branch at all.",
        body)]
    env = [
        ["**Load", "**Facility water", "**Pumps", "**Coolant supply", "**Peak case",
         "**Throttling"],
        ["90%", "26 °C", "100%", "33.2", "60.4", "no"],
        ["90%", "34 °C", "100%", "41.3", "69.5", "no"],
        ["90%", "42 °C", "100%", "49.5", "78.6", "no"],
        ["100%", "42 °C", "80%", "50.0", "84.6", "no"],
        ["100%", "42 °C", "55%", "50.1", "**91.1", "**yes"],
        ["60%", "42 °C", "55%", "48.4", "77.6", "no"],
    ]
    st += [table(env, [W * .11, W * .21, W * .13, W * .21, W * .17, W * .17],
                 align_right=(3, 4))]
    st += [Spacer(1, 6)]
    st += [liquid_figure("L4_envelope.png",
        "<b>Figure 5.</b> Peak chip case temperature at full load across the warm-water "
        "band, by pump speed. The three curves are nearly parallel, which is the point: "
        "pump speed shifts the whole envelope, so a hall can trade pumping energy "
        "against how warm its facility water is allowed to run.")]

    # ---------------- what it changes ----------------
    st += [Paragraph("6.  What this changes for the paper", h1)]
    items = [
        ["**#", "**Consequence"],
        ["1", "<b>The accuracy experiment has room again.</b> The air hall's 0.995 "
              "neighbour correlation capped what any spatial model could add. Here the "
              "most strongly coupled racks are ones a distance-based model would not "
              "select at all."],
        ["2", "<b>The edge-shuffle ablation becomes a real test.</b> Shuffling edges in "
              "the air hall was expected to cost almost nothing. Here it should destroy "
              "the model, because the edges carry the plumbing and nothing else encodes "
              "it."],
        ["3", "<b>The directed-edge ablation is no longer cosmetic.</b> Manifold "
              "cross-talk runs strictly upstream to downstream, so reversing an edge is "
              "physically wrong rather than merely uninformative."],
        ["4", "<b>Placement gains a sharper lever.</b> Where to put a job becomes partly "
              "a question of which loop has headroom, which is invisible to any "
              "floor-plan heuristic."],
    ]
    st += [table(items, [W * .05, W * .95])]

    # ---------------- limitations ----------------
    st += [Paragraph("7.  Before this becomes a claim", h1)]
    st += [boxed([Paragraph(
        "Four things should happen before any of Section 4 appears in a paper. The first "
        "two address the circularity in Section 1 directly.", body)], BLACK)]
    st += [Spacer(1, 5)]
    todo = [
        ["**#", "**Action", "**Why"],
        ["1", "Model the shared facility water loop",
         "Cross-CDU coupling becomes small but non-zero rather than stipulated zero. "
         "Cheapest fix, and it removes the worst of the circularity."],
        ["2", "Sweep manifold_crosstalk",
         "If the graph model's advantage exists at 0.16 and vanishes at 0.05, the result "
         "is an artefact of a number with no source."],
        ["3", "Anchor the CDU curve to a datasheet",
         "Vendors publish approach-against-load curves. This converts an invented "
         "parameter into a cited one."],
        ["4", "State the framing precisely",
         "Not &ldquo;evaluated on a liquid-cooled data hall&rdquo; but &ldquo;evaluated "
         "on a parametric direct-to-chip model whose coupling structure is imposed by "
         "construction&rdquo;."],
    ]
    st += [table(todo, [W * .05, W * .33, W * .62])]
    st += [Spacer(1, 8)]
    st += [Paragraph("Known limitations of the model as it stands", h2)]
    st += [Paragraph(
        "Racks hang in parallel off a manifold rather than in a series cascade; immersion "
        "tanks and some rear-door designs are genuinely serial and would show stronger "
        "ordering still. Flow is set by manifold position rather than solved "
        "hydraulically, so a rack drawing more does not starve its neighbours. "
        "Single-phase only, with no two-phase or evaporative cold plates. The CDU "
        "approach is a smooth ramp in duty rather than a heat-exchanger effectiveness "
        "curve with a real NTU.", body)]

    doc.build(st)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p.relative_to(REPO)}  ({p.stat().st_size/1e6:.2f} MB)")

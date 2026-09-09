"""Build the PDF status report.

Set entirely in Times New Roman. The body is black on white with hairline rules and a
single grey for secondary text; every other colour in the document is inside a figure.

    uv run python scripts/make_pdf_report.py
"""
from __future__ import annotations

import pathlib
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image,
                                KeepTogether, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

REPO = pathlib.Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures" / "report"
OUT = REPO / "results" / "twin-build-report.pdf"

BLACK = colors.HexColor("#000000")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#000000")
FAINT = colors.HexColor("#BBBBBB")

FONTS = pathlib.Path("/System/Library/Fonts/Supplemental")


def register_fonts() -> str:
    """Register real Times New Roman if the system has it; fall back to the built-in
    PostScript Times, which is metrically the same face."""
    try:
        pdfmetrics.registerFont(TTFont("TimesNR", FONTS / "Times New Roman.ttf"))
        pdfmetrics.registerFont(TTFont("TimesNR-Bold", FONTS / "Times New Roman Bold.ttf"))
        pdfmetrics.registerFont(TTFont("TimesNR-Italic", FONTS / "Times New Roman Italic.ttf"))
        pdfmetrics.registerFont(
            TTFont("TimesNR-BoldItalic", FONTS / "Times New Roman Bold Italic.ttf"))
        pdfmetrics.registerFontFamily(
            "TimesNR", normal="TimesNR", bold="TimesNR-Bold",
            italic="TimesNR-Italic", boldItalic="TimesNR-BoldItalic")
        return "TimesNR"
    except Exception:
        return "Times-Roman"


F = register_fonts()
FB = "TimesNR-Bold" if F == "TimesNR" else "Times-Bold"
FI = "TimesNR-Italic" if F == "TimesNR" else "Times-Italic"

S = getSampleStyleSheet()
body = ParagraphStyle("body", parent=S["Normal"], fontName=F, fontSize=9.6,
                      leading=13.4, alignment=TA_JUSTIFY, spaceAfter=6,
                      textColor=BLACK)
lead = ParagraphStyle("lead", parent=body, fontSize=10.4, leading=14.6, spaceAfter=8)
h1 = ParagraphStyle("h1", parent=S["Normal"], fontName=FB, fontSize=13.5, leading=16,
                    spaceBefore=16, spaceAfter=5, textColor=BLACK)
h2 = ParagraphStyle("h2", parent=S["Normal"], fontName=FB, fontSize=10.4, leading=13,
                    spaceBefore=10, spaceAfter=3, textColor=BLACK)
small = ParagraphStyle("small", parent=body, fontSize=8.4, leading=11.2,
                       textColor=GREY, spaceAfter=5)
cap = ParagraphStyle("cap", parent=body, fontSize=8.2, leading=11, textColor=GREY,
                     alignment=0, spaceBefore=3, spaceAfter=12)
title = ParagraphStyle("title", parent=S["Normal"], fontName=FB, fontSize=19,
                       leading=22, textColor=BLACK, spaceAfter=3)
sub = ParagraphStyle("sub", parent=S["Normal"], fontName=FI, fontSize=10.6,
                     leading=14, textColor=BLACK, spaceAfter=10)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11, alignment=0,
                      spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName=FB)


def rule(w=0.9, colour=RULE, space=6):
    return HRFlowable(width="100%", thickness=w, color=colour,
                      spaceBefore=space, spaceAfter=space)


def figure_parts(name, caption, width=138 * mm):
    """[image, caption] as plain flowables, safe to nest inside a table cell."""
    p = FIG / name
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        w, h = im.size
    return [Image(str(p), width=width, height=width * h / w), Paragraph(caption, cap)]


def figure(name, caption, width=138 * mm):
    return KeepTogether(figure_parts(name, caption, width))


def table(rows, widths, align_right=(), header=True, font_size=8.4):
    data = []
    for i, r in enumerate(rows):
        out = []
        for j, c in enumerate(r):
            st = cellb if (i == 0 and header) else cell
            if isinstance(c, str) and c.startswith("**"):
                st, c = cellb, c[2:]
            out.append(Paragraph(c, st))
        data.append(out)
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        # Every cell holds a Paragraph, but reportlab still registers its default
        # Helvetica on the page unless the table is told otherwise. The document is
        # meant to be Times throughout.
        ("FONTNAME", (0, 0), (-1, -1), F),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, FAINT),
    ]
    if header:
        style += [("LINEABOVE", (0, 0), (-1, 0), 0.9, RULE),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE)]
    style += [("LINEBELOW", (0, -1), (-1, -1), 0.9, RULE)]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 7.6)
    canvas.setFillColor(GREY)
    canvas.drawString(20 * mm, 12 * mm,
                      "Digital twin build report  ·  commit d829879  ·  9 September 2026")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(FAINT)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm,
                          title="Digital twin build report",
                          author="Arnav Bharadwaj")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    W = doc.width
    st = []

    # ---------------- title ----------------
    st += [
        Paragraph("Digital Twin-Assisted Workload Allocation and "
                  "Cooling Optimisation", title),
        Paragraph("Build report: status, findings, and data provenance", sub),
        rule(1.1, RULE, 4),
        Paragraph(
            "This report records the state of the research codebase, in the order the "
            "work happened, together with the origin of every input. It is written to be "
            "checked rather than trusted: each claim below is a number produced by a "
            "script in the repository and logged to <font name='%s'>results/runs.csv</font> "
            "with its configuration hash, seed and git commit." % FI, lead),
        rule(0.4, FAINT, 4),
    ]

    st += [table([
        ["**Phases complete", "**Tests", "**Simulated time", "**Logged runs", "**Commits"],
        ["0, 0b, 1, 2, 3", "144 passing", "1,500 hours", "173 rows", "7 on main"],
    ], [W * .24, W * .17, W * .21, W * .18, W * .20], font_size=8.6), Spacer(1, 4)]

    # ---------------- 1. summary ----------------
    st += [Paragraph("1.  Summary", h1)]
    st += [Paragraph(
        "The project set out to build a digital twin of a data hall whose internal state is "
        "a directed graph, train a message-passing network to predict per-rack inlet "
        "temperature, and drive two controllers from that forecast. Phases 0 through 3 are "
        "complete. Phases 4 (online calibration and drift), 5 (the two controllers) and 6 "
        "(the ablations) have not been started.", body)]
    st += [Paragraph(
        "Two results reshaped the plan. The first is that the chosen simulator has no "
        "thermal physics at all, which forced its heat model to be rewritten before any "
        "comparison could be meaningful. The second is that on the resulting twin, "
        "neighbouring racks' temperatures correlate at 0.995, so spatial context adds far "
        "less to single-rack accuracy than the proposal assumed. The graph model still "
        "wins at every horizon, and it is the only model of the five that transfers to an "
        "unseen hall without refitting &mdash; which is where the paper's weight should now sit.",
        body)]

    # ---------------- 2. chronology ----------------
    st += [Paragraph("2.  Chronology", h1)]
    st += [Paragraph(
        "Ten entries, in the order they occurred. The two marked as pivots changed the "
        "direction of the work.", small)]

    chron = [
        ["**#", "**Stage", "**Commit", "**Outcome", "**What happened"],
        ["01", "Setup", "&mdash;", "done",
         "Machine is an Apple M2 carrying an x86_64 Homebrew and x86_64 Python. PyTorch has "
         "shipped no x86_64 macOS wheels since 2.2.2. Installed native arm64 uv "
         "(checksum-verified), pinned CPython 3.11.16 arm64, torch 2.14.0 with MPS."],
        ["02", "Phase 0", "8aba8b1", "**gate failed",
         "SustainDC computes inlet temperature as a JSON constant plus the CRAC setpoint. "
         "Rack power never enters. A 14.8 kW swing on one rack moves every rack by exactly "
         "0.000e+00 K. The influence matrix is the zero matrix, including its diagonal."],
        ["03", "Decision", "&mdash;", "Option A",
         "Keep SustainDC for its HVAC power chain and exogenous data; replace the inlet "
         "computation; state in the paper that the thermal ground truth is ours."],
        ["04", "Phase 0b", "00bb5a3", "verified",
         "Heat-recirculation kernel, T = T<sub>supply</sub> + D·P with "
         "D = K<super>-1</super>[(I − A<super>T</super>)<super>-1</super> − I]. "
         "Self-response 0.292 K, cross-rack up to 0.255 K, 29.7% of influence beyond one "
         "hop, setpoint gain 0.878–0.923 rather than pinned at 1."],
        ["05", "Phase 1", "6957b99", "gate passed",
         "1,500 h of trajectories across three halls at 30 s. Alibaba PAI jobs replayed, four "
         "placement policies, stepped cooling setpoints. Three calibration errors caught by "
         "measuring the output rather than trusting the config."],
        ["06", "Phase 2", "5e43bd6", "**finding",
         "LightGBM with 2, 4 and 6 neighbours scores identically to three decimals. Own-rack "
         "features give R² = 0.825 on the 300 s target; adding every neighbour feature gives "
         "0.827. Neighbour temperature is nearly redundant."],
        ["07", "Correction", "46545dd", "task fixed",
         "73.6% of the 300 s target variance came from setpoint changes that had not happened "
         "yet, against 4.3% from the state at t. The control trajectory became an input. "
         "LightGBM went from 0.653 to 0.188 K at 300 s."],
        ["08", "Phase 3", "d829879", "complete",
         "Graph model trained on hall_a only; hall_b and hall_c evaluated zero-shot with the "
         "same weights and the same normaliser. Wins at every horizon and holds its skill "
         "on unseen halls."],
        ["09", "Phase 2b", "&mdash;", "running",
         "Baselines refit on hall_b and hall_c, so the transfer result has a bar to beat. "
         "Phase 2 covered hall_a only."],
        ["10", "Phases 4–6", "&mdash;", "not started",
         "Calibration and drift injection; the placement and cooling controllers; the five "
         "ablations."],
    ]
    st += [table(chron, [W * .04, W * .10, W * .10, W * .11, W * .65])]

    # ---------------- 3. phase 0 ----------------
    st += [Paragraph("3.  The Phase 0 finding", h1)]
    st += [Paragraph(
        "Rack inlet temperature is computed in exactly one place in the whole SustainDC "
        "repository, <font name='%s'>envs/datacenter.py:279</font>:" % FI, body)]
    st += [Paragraph(
        "<font name='%s'>rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint</font>"
        % FI,
        ParagraphStyle("code", parent=body, leftIndent=14, spaceBefore=2, spaceAfter=6,
                       alignment=0))]
    st += [Paragraph(
        "The approach temperature is a constant read from JSON at startup and never "
        "recomputed. No rack's power enters &mdash; not a neighbour's, not the rack's own. The "
        "consequence is that every model in the planned comparison would have tied at machine "
        "precision, and every ablation would have been null by construction. Two further "
        "blockers were found while probing: the gym wrapper hardcodes one scalar utilisation "
        "for all racks, so placement was inexpressible, and three of the four shipped hall "
        "configurations do not load.", body)]
    st += [figure("fig1_coupling.png",
                  "<b>Figure 1.</b> Response of every rack's inlet temperature to a 14.8 kW "
                  "perturbation at one rack. SustainDC's values are exactly zero at every "
                  "distance, including zero distance; its curve terminates at distance 7 "
                  "because its default hall holds only 20 racks. The replacement kernel decays "
                  "smoothly across the hall.", 138 * mm)]

    # ---------------- 4. the replacement ----------------
    st += [Paragraph("4.  The replacement thermal model", h1)]
    st += [Paragraph(
        "With <font name='%s'>a<sub>ij</sub></font> the fraction of rack <i>i</i>'s exhaust "
        "drawn into rack <i>j</i>'s inlet and K the diagonal matrix of rack thermal mass "
        "flows, the mixing energy balance at each inlet rearranges to "
        "<font name='%s'>T = T<sub>supply</sub> + D·P</font>, with "
        "<font name='%s'>D = K<super>-1</super>[(I − A<super>T</super>)<super>-1</super> − I]</font>. The matrix inverse expands as a "
        "Neumann series in A<super>T</super>, so multi-hop coupling emerges even though A is "
        "local: term "
        "<i>m</i> is air that passed through <i>m</i> racks before arriving. A is built from "
        "the floor plan alone &mdash; distance, rows crossed, and direction relative to the "
        "return-air path &mdash; never from the kernel the model is being asked to learn."
        % (FI, FI, FI), body)]

    tw = W * 0.47
    left = figure_parts("fig2_hops.png",
                        "<b>Figure 2.</b> Share of total influence arriving after <i>m</i> "
                        "hops. Roughly 30% arrives beyond the first, which is what justifies "
                        "three message-passing layers.", tw - 6 * mm)
    right = figure_parts("fig3_redundancy.png",
                         "<b>Figure 3.</b> A rack's own inlet temperature against the mean of "
                         "its six nearest racks, hall_a. The dashed line is equality.",
                         tw - 6 * mm)
    pair = Table([[left, right]], colWidths=[tw, tw], hAlign="LEFT")
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (0, 0), 8)]))
    st += [pair]

    st += [Paragraph(
        "The coupling is state-dependent, not a fixed matrix. K moves with rack fan speed, and "
        "the leakage fraction rises when the racks draw more air than the CRACs supply, so a "
        "cooling action reshapes the coupling rather than merely shifting the temperature "
        "field. SustainDC's setpoint gain was exactly 1.000000 at every rack, a rigid "
        "translation; the replacement's ranges from 0.878 to 0.923 and varies by rack.", body)]

    st += [Paragraph("A modelling bug worth recording", h2)]
    st += [Paragraph(
        "Normalising each source's leakage by geometry alone leaves a rack's recirculated "
        "intake fixed while its total intake grows with fan speed. Ramping its fans then "
        "dilutes its own inlet, and loading a rack from 50% to 100% utilisation <i>cooled</i> "
        "it by 0.79 K while its neighbours warmed by 0.08 K. A placement controller trained "
        "against that would have learned to stack load onto the hottest racks. Weighting each "
        "target's share of the recirculating air by its own thermal mass flow removes the "
        "artefact; two regression tests hold it removed.", body)]

    # ---------------- 5. dataset ----------------
    st += [Paragraph("5.  The dataset", h1)]
    st += [Paragraph(
        "Real Alibaba PAI GPU jobs are replayed onto racks under four placement policies, "
        "including a deliberately bad corner-stacking policy so that the dataset contains hot "
        "states. Cooling setpoints are held for 5 to 20 minutes and then stepped, as a real "
        "CRAC controller behaves. 500 hours per hall at 30-second resolution, split by episode "
        "so that no input window crosses a boundary and every training timestep precedes every "
        "test timestep.", body)]
    st += [figure("fig4_gate.png",
                  "<b>Figure 4.</b> Distribution of rack inlet temperature for the three halls "
                  "against the ASHRAE 2021 class A1 limits. The Phase 1 gate required at least "
                  "2% of timesteps to breach the recommended limit; the halls reach 18.9%, "
                  "18.8% and 19.3% respectively, and none exceeds the allowable limit "
                  "materially. hall_c runs colder on average but reaches further, which is the "
                  "signature of its uncontained layout.", 138 * mm)]

    # ---------------- 6. results ----------------
    st += [Paragraph("6.  Results", h1)]
    st += [Paragraph("6.1  Accuracy on hall_a", h2)]
    st += [Paragraph(
        "Test RMSE of the temperature delta, in kelvin, over five seeds. The graph model wins "
        "at every horizon, with the largest margin at 60 s. That shape is consistent with the "
        "graph mattering most once heat has had time to propagate between racks but before the "
        "hall settles toward its steady state.", body)]

    acc = [
        ["**Model", "**Spatial input", "**30 s", "**60 s", "**300 s"],
        ["Persistence", "none", "0.1057", "0.1908", "0.6367"],
        ["RC network", "graph edges", "0.1682", "0.1978", "0.4833"],
        ["Per-rack LSTM", "none", "0.0272", "0.0598", "0.2245"],
        ["LightGBM <i>k</i>=4", "4 nearest racks", "0.0433", "0.0606", "0.1884"],
        ["**Graph model", "**message passing", "**0.0244", "**0.0434", "**0.1677"],
    ]
    st += [table(acc, [W * .22, W * .26, W * .17, W * .17, W * .18], align_right=(2, 3, 4))]
    st += [Spacer(1, 3), Paragraph(
        "Margin over the best baseline: 11% at 30 s, 27% at 60 s, 11% at 300 s. Seed spread is "
        "±0.003 to ±0.004, so the ordering is stable.", small)]

    st += [Paragraph("6.2  Zero-shot transfer", h2)]
    st += [Paragraph(
        "The model is trained on hall_a and applied unchanged to the other two halls, carrying "
        "hall_a's normalisation statistics with it; those statistics are never recomputed on a "
        "target hall, which would be leakage. Results are given as skill, the fraction of the "
        "trivial predictor's error removed, because the three halls have different target "
        "variance and raw RMSE is not comparable across them.", body)]
    tr = [
        ["**Hall", "**Relationship to hall_a", "**30 s", "**60 s", "**300 s"],
        ["hall_a", "in-domain, 200 racks", "76.0%", "75.0%", "70.8%"],
        ["hall_b", "140 racks, same aisle pattern", "81.6%", "78.9%", "71.7%"],
        ["hall_c", "uncontained layout, moved CRACs", "71.6%", "74.4%", "68.2%"],
    ]
    st += [table(tr, [W * .13, W * .37, W * .16, W * .16, W * .18], align_right=(2, 3, 4))]
    st += [Spacer(1, 6)]
    st += [figure("fig5_accuracy.png",
                  "<b>Figure 5.</b> Accuracy on hall_a by prediction horizon. Error bars are "
                  "one standard deviation over five seeds.", 138 * mm)]
    st += [figure("fig6_transfer.png",
                  "<b>Figure 6.</b> Skill against each hall's own persistence baseline. The "
                  "transferred model removes as much of the trivial predictor's error on halls "
                  "it has never seen as it does in-domain.", 138 * mm)]
    st += [Paragraph(
        "Transfer holds. On hall_b the transferred model removes more of persistence's error "
        "than it does in-domain, because that hall's larger target variance makes the trivial "
        "predictor worse. Layout transfer to hall_c costs roughly four skill points; scale "
        "transfer costs nothing. LightGBM's <i>k</i>-nearest features and the RC network's "
        "per-rack parameters cannot transfer at all, which is why this is the experiment with "
        "room in it.", body)]

    # ---------------- 7. provenance ----------------
    st += [Paragraph("7.  Data provenance", h1)]
    st += [Paragraph(
        "After Phase 0, the inlet model, the thermal dynamics, the rack power curves and the "
        "hall geometry are all ours. What remains of SustainDC is its HVAC power chain and its "
        "exogenous data series. The paper cannot describe its thermal results as evaluated on "
        "SustainDC; the honest framing is that the twin is our own physics-based recirculation "
        "model, that SustainDC supplies the plant-side power chain, and that the thermal "
        "ground truth is therefore synthetic.", body)]
    prov = [
        ["**Component", "**Origin", "**Identifier"],
        ["Rack inlet temperature", "ours", "src/twin/recirculation.py; formulation after Tang et al."],
        ["Thermal dynamics", "ours", "src/twin/thermal.py; first-order lag, τ = 60 s"],
        ["Rack power and airflow", "ours", "src/twin/power.py; SustainDC's curves rejected as hand-tuned"],
        ["Hall geometry and aisles", "ours", "src/twin/geometry.py; SustainDC has no geometry"],
        ["HVAC chain: chiller, tower, pumps", "SustainDC", "HewlettPackard/dc-rl @ a92b4755aca560e34a98d14028dda629eb968482"],
        ["Weather and carbon intensity", "SustainDC", "same checkout, pinned by scripts/bootstrap_external.sh"],
        ["Workload trace", "Alibaba", "alibaba/clusterdata, cluster-trace-gpu-v2020, pai_job_duration_estimate_100K.csv: 100,000 jobs over 289.8 h, 12.7 MB"],
        ["Thermal limits", "ASHRAE 2021", "class A1: 27 °C recommended, 32 °C allowable"],
        ["Trajectories", "generated", "data/trajectories/: hall_a 86 MB, hall_b 61 MB, hall_c 83 MB; regenerable from config and seed"],
        ["All results", "generated", "results/runs.csv: 173 rows, each with config hash, seed and git SHA"],
    ]
    st += [table(prov, [W * .26, W * .14, W * .60])]
    st += [Spacer(1, 3), Paragraph(
        "The workload trace is bootstrapped rather than tiled: jobs are drawn independently "
        "from the empirical pool with arrivals following the trace's own hour-of-day profile. "
        "Tiling was rejected because the split is by time, so an exact repeat would place the "
        "same jobs in both training and test. It is a resampling of real data and should be "
        "described as such.", small)]

    # ---------------- 8. open items ----------------
    st += [Paragraph("8.  Open items before submission", h1)]
    items = [
        ["**#", "**Item", "**Why it matters"],
        ["1", "Re-run Phase 3 with memory headroom",
         "The graph model's headline number came from a network of hidden width 32 with two "
         "attention heads, trained on 210,000 rows against LightGBM's 400,000. This machine has "
         "8.6 GB of unified memory with several gigabytes already swapped; MPS measured about "
         "70× slower than CPU once trajectories were resident, and activation memory hits a "
         "swap cliff that moves with the machine's state. The confound runs against the graph "
         "model, which won regardless, but it should not have to be explained away."],
        ["2", "Baselines on hall_b and hall_c",
         "Running at the time of writing. Without them the transfer table shows the graph model "
         "holding its skill on unseen halls with nothing measured on those halls to compare "
         "against."],
        ["3", "Phase 6 before Phase 5",
         "The edge-shuffle ablation directly tests the Phase 2 prediction that the graph is "
         "close to decorative. It is cheap, and it is the result most likely to change how the "
         "paper is framed. Building two controllers on top of a graph that carries little "
         "information would be an expensive way to discover this."],
        ["4", "State the kernel limitation plainly",
         "The 0.995 neighbour correlation is a property of our recirculation kernel's decay "
         "length. A hall with tighter, more local recirculation would show more spatial signal "
         "and give a graph model more room. This belongs in the limitations section."],
    ]
    st += [table(items, [W * .04, W * .28, W * .68])]

    doc.build(st)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p.relative_to(REPO)}  ({p.stat().st_size/1e6:.2f} MB)")

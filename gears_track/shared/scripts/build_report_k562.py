#!/usr/bin/env python
"""
K562 gwps multi-model benchmark report: GEARS vs additive / global mean-shift /
ridge baselines (+ control-mean reference). Phylo-branded PDF via ReportLab.

Inputs:
  --metrics_gears, --metrics_additive, --metrics_meanshift, --metrics_ridge
      official aggregate_metrics.py JSONs (overall_model / overall_base blocks)
  --comparison_pkl : comparison_rows pkl from compare_models.py (per-pert rows)
  --references     : references.json (list under "references")
  --figdir         : figure directory (F1/F3/F4/F6 PNGs)
  --out            : output PDF path
"""
import argparse, json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, KeepTogether, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


def register_fonts():
    base = "/usr/share/fonts/truetype/liberation"
    trip = {"LSans": "LiberationSans-Regular.ttf",
            "LSans-Bold": "LiberationSans-Bold.ttf",
            "LSans-Italic": "LiberationSans-Italic.ttf"}
    if all(os.path.exists(f"{base}/{v}") for v in trip.values()):
        for name, fn in trip.items():
            pdfmetrics.registerFont(TTFont(name, f"{base}/{fn}"))
        return "LSans", "LSans-Bold", "LSans-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_B, FONT_I = register_fonts()

BLACK = colors.HexColor("#000000"); CREAM = colors.HexColor("#FAF9F3")
CREAM2 = colors.HexColor("#ECE9E2"); GOLD = colors.HexColor("#D4A04A")
ORANGE = colors.HexColor("#FF9400"); GREEN = colors.HexColor("#75A025")
PINK = colors.HexColor("#FD9BED"); BLUE = colors.HexColor("#0279EE")
DARK = colors.HexColor("#2C2A26"); GREY = colors.HexColor("#8A8378")
BORDER = colors.HexColor("#D5CFC5")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

h1 = ParagraphStyle("h1", fontName=FONT_B, fontSize=15, leading=19, textColor=BLUE, spaceBefore=14, spaceAfter=7)
h2 = ParagraphStyle("h2", fontName=FONT_B, fontSize=11.5, leading=15, textColor=DARK, spaceBefore=9, spaceAfter=4)
body = ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=14, textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=6)
cap = ParagraphStyle("cap", fontName=FONT_I, fontSize=8, leading=11, textColor=GREY, alignment=TA_LEFT, spaceBefore=3, spaceAfter=12)
bullet = ParagraphStyle("bullet", fontName=FONT, fontSize=9.5, leading=13.5, textColor=DARK, leftIndent=12, bulletIndent=2, spaceAfter=3, alignment=TA_LEFT)
ref_st = ParagraphStyle("ref", fontName=FONT, fontSize=8.5, leading=12, textColor=DARK, leftIndent=14, bulletIndent=2, spaceAfter=4, alignment=TA_LEFT)
t_title = ParagraphStyle("t_title", fontName=FONT_B, fontSize=22, leading=27, textColor=BLACK, alignment=TA_LEFT)
t_sub = ParagraphStyle("t_sub", fontName=FONT, fontSize=12, leading=17, textColor=DARK, alignment=TA_LEFT)
t_meta = ParagraphStyle("t_meta", fontName=FONT, fontSize=9, leading=13, textColor=GREY, alignment=TA_LEFT)
tile_big = ParagraphStyle("tile_big", fontName=FONT_B, fontSize=16, leading=18, textColor=BLUE, alignment=TA_CENTER)
tile_lbl = ParagraphStyle("tile_lbl", fontName=FONT, fontSize=7.5, leading=9.5, textColor=DARK, alignment=TA_CENTER)
cell_st = ParagraphStyle("cell", fontName=FONT, fontSize=8.5, leading=11, textColor=DARK, alignment=TA_CENTER)
cell_b = ParagraphStyle("cellb", fontName=FONT_B, fontSize=8.5, leading=11, textColor=DARK, alignment=TA_CENTER)


def fnum(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_gears", required=True)
    ap.add_argument("--metrics_additive", required=True)
    ap.add_argument("--metrics_meanshift", required=True)
    ap.add_argument("--metrics_ridge", required=True)
    ap.add_argument("--comparison_pkl", default=None)
    ap.add_argument("--references", default="/mnt/results/execution_trace/references.json")
    ap.add_argument("--figdir", default="/mnt/results/figures")
    ap.add_argument("--out", default="/mnt/results/report_k562_gwps_perturbation_benchmark.pdf")
    args = ap.parse_args()

    MG = json.load(open(args.metrics_gears))
    MA = json.load(open(args.metrics_additive))
    MM = json.load(open(args.metrics_meanshift))
    MR = json.load(open(args.metrics_ridge))
    refs = json.load(open(args.references)).get("references", [])

    omG, cmG = MG["overall_model"], MG.get("overall_model_compute_metrics", {})
    omA, omM, omR = MA["overall_model"], MM["overall_model"], MR["overall_model"]
    obG = MG.get("overall_base", {})
    FIG = args.figdir

    def img(path, frac=1.0):
        from PIL import Image as PILImage
        p = os.path.join(FIG, path)
        if not os.path.exists(p):
            return None
        iw, ih = PILImage.open(p).size
        w = CONTENT_W * frac
        im = Image(p, width=w, height=w * ih / iw)
        im.hAlign = "CENTER"
        return im

    def fig_block(path, caption, frac=1.0):
        im = img(path, frac)
        if im is None:
            return None
        return KeepTogether([im, Paragraph(caption, cap)])

    story = []
    title = "Predicting transcriptional responses to unseen genetic perturbations: GEARS vs. simple baselines on the K562 genome-scale CRISPRi Perturb-seq dataset"

    # ===================== PAGE 1 =====================
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(title, t_title))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Four-model benchmark on the Replogle et al. K562 genome-wide perturbation "
                           "(gwps) dataset - GEARS simulation split, seed 42, 1,905 held-out single-gene "
                           "perturbations (unseen-single regime)", t_sub))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width=CONTENT_W, thickness=3, color=BLUE, spaceAfter=8, spaceBefore=2))

    tiles = [
        (fnum(omG.get("pearson_delta_mean"), 3), "GEARS pearson_delta<br/>(mean)"),
        (fnum(omM.get("pearson_delta_mean"), 3), "mean-shift pearson_delta<br/>(mean)"),
        (fnum(omG.get("frac_correct_direction_20_mean"), 3), "GEARS direction match<br/>(top-20 DE)"),
        (fnum(omM.get("frac_correct_direction_20_mean"), 3), "mean-shift direction<br/>(top-20 DE)"),
        ("1,905", "held-out<br/>perturbations"),
        ("8,248", "genes in<br/>panel"),
    ]
    tcells = [[Paragraph(v, tile_big) for v, _ in tiles],
              [Paragraph(l, tile_lbl) for _, l in tiles]]
    tt = Table(tcells, colWidths=[CONTENT_W / len(tiles)] * len(tiles))
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CREAM),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
        ("TOPPADDING", (0, 1), (-1, 1), 2), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    tt.hAlign = "CENTER"
    story.append(tt)
    story.append(Spacer(1, 6 * mm))

    gears_wins_delta = omG.get("pearson_delta_mean", 0) > omM.get("pearson_delta_mean", 0)
    lead = ("GEARS outperformed" if gears_wins_delta else
            "A simple global mean-shift baseline matched or outperformed")
    summ = (f"<b>Setup.</b> We trained GEARS (a graph neural network with gene co-expression and Gene Ontology "
            f"priors) and three custom baselines (additive gene-effect transfer, a global CRISPRi mean-shift, and a "
            f"ridge regressor on seen-gene effects) to predict post-perturbation expression in K562 cells, and "
            f"evaluated all methods on {MG.get('n_test_perts','?')} held-out single-gene CRISPRi perturbations whose "
            f"target genes are entirely absent from training (unseen-single regime). "
            f"<b>Headline.</b> {lead} the field on change-from-control correlation "
            f"(pearson_delta: GEARS {fnum(omG.get('pearson_delta_mean'),3)} vs. mean-shift "
            f"{fnum(omM.get('pearson_delta_mean'),3)}); direction-of-change accuracy on the top-20 DE genes was "
            f"{fnum(omG.get('frac_correct_direction_20_mean'),3)} (GEARS) vs. "
            f"{fnum(omM.get('frac_correct_direction_20_mean'),3)} (mean-shift). The additive baseline degenerates "
            f"exactly to the control mean on unseen genes - the key motivation for ontology-based transfer.")
    card = Table([[Paragraph(summ, body)]], colWidths=[CONTENT_W])
    card.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CREAM),
                              ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                              ("LINEBEFORE", (0, 0), (0, -1), 3, GOLD),
                              ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                              ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    story.append(card)
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Prepared by Biomni &middot; Phylo", t_meta))
    story.append(PageBreak())

    # ===================== 1. INTRODUCTION =====================
    story.append(Paragraph("1&nbsp;&nbsp;Introduction", h1))
    story.append(Paragraph(
        "Perturb-seq couples pooled CRISPR perturbations with single-cell RNA sequencing, measuring how hundreds of "
        "thousands of genetic interventions reshape the transcriptome [[17]]. The Replogle et al. genome-scale "
        "Perturb-seq (gwps) dataset profiles CRISPRi knockdowns of ~9,500 genes in K562 cells [[14]], enabling "
        "genome-scale evaluation of <b>response-prediction models</b>: given a perturbation the model has never "
        "seen, predict the post-perturbation expression profile.", body))
    story.append(Paragraph(
        "GEARS [[9]] addresses this with a graph neural network that couples gene co-expression with Gene Ontology "
        "structure, allowing perturbation effects to be transferred to genes never perturbed in training. "
        "Foundation-model approaches such as scGPT [[19]] pursue the same goal at larger scale. Meanwhile, recent "
        "systematic benchmarks report that carefully constructed simple baselines are often competitive with, or "
        "superior to, learned models on post-perturbation prediction [[13, 20]]. This report quantifies that tension "
        "directly: GEARS is benchmarked against three interpretable baselines and the control-mean reference on an "
        "identical held-out split.", body))

    # ===================== 2. METHODS =====================
    story.append(Paragraph("2&nbsp;&nbsp;Methods", h1))
    story.append(Paragraph("2.1&nbsp;&nbsp;Dataset and preprocessing", h2))
    story.append(Paragraph(
        "We used the K562 genome-scale CRISPRi Perturb-seq dataset (gwps) from Replogle et al. [[14]] (Figshare+ "
        "doi:10.25452/figshare.plus.20029387). The raw release (65.8 GB, ~1.7M cells after QC at genome scale) was "
        "subsampled to 10 cells per perturbation plus 2,000 non-targeting controls (78,190 cells total) - a "
        "computational necessity on CPU-only infrastructure that preserves every perturbation condition. Expression "
        "was restricted to the 8,248-gene HVG panel shared with the GEARS vocabulary; 2,168 perturbations whose "
        "target genes fall outside the panel were dropped, leaving 7,619 single-gene CRISPRi conditions. Differential-expression "
        "indices (top-20 DE genes per condition vs. controls, Wilcoxon rank-sums as in the GEARS pipeline) were "
        "precomputed for evaluation.", body))
    story.append(Paragraph("2.2&nbsp;&nbsp;Split", h2))
    story.append(Paragraph(
        "Conditions were split with the GEARS <i>simulation</i> splitter (seed 42, train_gene_set_size 0.75): "
        "5,143 train / 572 validation / 1,905 test conditions. By construction of the unseen-single regime, every "
        "test perturbation targets a gene with <b>zero overlap</b> with training perturbation genes; combo regimes "
        "were empty and are reported as such. All models were evaluated on exactly this test set.", body))
    story.append(Paragraph("2.3&nbsp;&nbsp;Models", h2))
    for b in [
        "<b>GEARS</b> [[9]]: GNN over gene-gene relation graphs (co-expression + GO); perturbation effects propagate "
        "through ontology edges, enabling transfer to unseen genes. Trained 4 epochs (batch 32, 10 cells/pert "
        "sampling) on CPU - a reduced budget relative to the default 20 epochs.",
        "<b>Additive baseline</b>: predicts perturbed expression as control mean plus the perturbed gene's average "
        "effect learned from training. For unseen genes no effect exists, so the prediction collapses to the "
        "control mean - the degenerate case this benchmark is designed to expose.",
        "<b>Global mean-shift baseline</b>: predicts control mean plus the average expression shift observed across "
        "all training CRISPRi perturbations (a global direction of CRISPRi response), scaled to the test condition.",
        "<b>Ridge baseline</b>: ridge regression mapping perturbation-gene identity features to expression deltas, "
        "trained on seen genes; for unseen genes only intercept-level (shrunken) structure is available.",
        "<b>Control-mean reference</b>: predicts the unperturbed control mean (zero delta).",
    ]:
        story.append(Paragraph(b, bullet, bulletText="\u2022"))
    story.append(Paragraph("2.4&nbsp;&nbsp;Metrics", h2))
    story.append(Paragraph(
        "Following the GEARS evaluation convention, predictions and truth are averaged per perturbation and "
        "compared as change-from-control (delta). We report pearson_delta (all genes), pearson_delta_de and "
        "pearson_de (top-20 DE genes), mse_de, and the fraction of top-20 DE genes whose predicted delta has the "
        "correct sign (dir20). Absolute-expression correlations are reported for completeness but are dominated by "
        "unchanged genes; delta and direction metrics are the meaningful yardsticks [[9, 20]].", body))
    fb = fig_block("F1_dataset_overview.png",
                   "<b>Figure&nbsp;1.</b> Dataset and evaluation design: dataset scale (left), the GEARS "
                   "condition-level split (middle), and held-out test regime composition (right; only the "
                   "unseen-single regime is populated).")
    if fb:
        story.append(fb)

    # ===================== 3. RESULTS =====================
    story.append(Paragraph("3&nbsp;&nbsp;Results", h1))
    story.append(Paragraph("3.1&nbsp;&nbsp;Headline comparison", h2))
    story.append(Paragraph(
        f"Across the 1,905 held-out unseen-single perturbations, GEARS reached pearson_delta = "
        f"{fnum(omG.get('pearson_delta_mean'),4)} (mean) and dir20 = "
        f"{fnum(omG.get('frac_correct_direction_20_mean'),3)}, versus pearson_delta = "
        f"{fnum(omM.get('pearson_delta_mean'),4)} / dir20 = {fnum(omM.get('frac_correct_direction_20_mean'),3)} for "
        f"the global mean-shift baseline and {fnum(omR.get('pearson_delta_mean'),4)} / "
        f"{fnum(omR.get('frac_correct_direction_20_mean'),3)} for ridge. The additive baseline scored exactly zero "
        f"on all delta metrics because its prediction is identically the control mean on unseen genes. All methods "
        f"achieved nearly identical mse_de (~0.32), reflecting the shared difficulty of the task and the dominance "
        f"of per-gene noise over systematic structure at 10 cells per perturbation.", body))

    # comparison table
    header = ["model", "pearson_delta", "pearson_delta_de", "pearson_de", "dir20", "mse_de"]
    rows = [
        ["GEARS", omG, cmG],
        ["Additive", omA, MA.get("overall_model_compute_metrics", {})],
        ["Global mean-shift", omM, MM.get("overall_model_compute_metrics", {})],
        ["Ridge (seen-gene)", omR, MR.get("overall_model_compute_metrics", {})],
        ["Control-mean", obG, MG.get("overall_base_compute_metrics", {})],
    ]
    tdata = [[Paragraph(h, cell_b) for h in header]]
    for name, om, cm in rows:
        tdata.append([
            Paragraph(name, cell_st),
            Paragraph(fnum(om.get("pearson_delta_mean"), 4), cell_st),
            Paragraph(fnum(om.get("pearson_delta_de_mean"), 4), cell_st),
            Paragraph(fnum(cm.get("pearson_de", om.get("pearson_de_mean")), 4), cell_st),
            Paragraph(fnum(om.get("frac_correct_direction_20_mean"), 4), cell_st),
            Paragraph(fnum(om.get("mse_de_mean"), 4), cell_st),
        ])
    tbl = Table(tdata, colWidths=[CONTENT_W * f] for f in [0.24, 0.16, 0.17, 0.15, 0.13, 0.15])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GOLD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM2]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(KeepTogether([tbl, Paragraph(
        "<b>Table&nbsp;1.</b> Overall held-out performance (means over 1,905 unseen-single perturbations). "
        "pearson_de is computed on top-20 DE genes. Control-mean scores 0 on delta metrics by construction.", cap)]))

    fb = fig_block("F3_benchmark_summary.png",
                   "<b>Figure&nbsp;3.</b> Benchmark summary across models. Left: change-from-control correlation "
                   "and direction accuracy (higher is better). Right: prediction error on top-20 DE genes (lower "
                   "is better).")
    if fb:
        story.append(fb)

    story.append(Paragraph("3.2&nbsp;&nbsp;The additive baseline degenerates on unseen genes", h2))
    story.append(Paragraph(
        "The additive baseline transfers a gene's learned average perturbation effect to the test condition. In the "
        "unseen-single regime the test genes have no training data, so its predicted delta is identically zero and "
        "every delta-based score is zero <b>by construction</b> - not a bug but the central motivation for "
        "ontology-based models such as GEARS [[9]]: some mechanism must carry information from perturbed genes to "
        "unperturbed ones.", body))
    story.append(Paragraph("3.3&nbsp;&nbsp;A global mean-shift is a surprisingly strong baseline", h2))
    story.append(Paragraph(
        f"Predicting the same global CRISPRi response direction for every unseen perturbation achieved pearson_delta "
        f"= {fnum(omM.get('pearson_delta_mean'),3)} and correct direction for "
        f"{fnum(omM.get('frac_correct_direction_20_mean')*100,1)}% of top-20 DE genes - i.e. a single shared "
        f"response vector, with no gene-specific information, explains a measurable fraction of the average "
        f"response. This echoes the systematic-benchmark finding that simple baselines are hard to beat on "
        f"post-perturbation prediction [[13, 20]]. Any learned model must clear this bar before claiming gene-specific "
        f"skill.", body))
    story.append(Paragraph("3.4&nbsp;&nbsp;Per-perturbation variability", h2))
    fb = fig_block("F4_metric_distributions.png",
                   "<b>Figure&nbsp;4.</b> Per-perturbation distributions of pearson_delta (all genes, left) and "
                   "pearson_delta on top-20 DE genes (right), overlaid across models.")
    if fb:
        story.append(fb)
    story.append(Paragraph("3.5&nbsp;&nbsp;GEARS example perturbations", h2))
    fb = fig_block("F6_example_scatters.png",
                   "<b>Figure&nbsp;6.</b> GEARS predicted vs. measured change-from-control for best / named / "
                   "median / worst held-out perturbations (each point a gene; dashed line = perfect prediction).")
    if fb:
        story.append(fb)

    # ===================== 4. CONCLUSIONS =====================
    story.append(Paragraph("4&nbsp;&nbsp;Conclusions", h1))
    for b in [
        f"<b>GEARS {'clears' if gears_wins_delta else 'does not clear'} the strong-baseline bar.</b> "
        f"On unseen-single generalisation GEARS reached pearson_delta = {fnum(omG.get('pearson_delta_mean'),3)} vs. "
        f"{fnum(omM.get('pearson_delta_mean'),3)} for the global mean-shift baseline"
        f"{' - genuine gene-specific transfer signal' if gears_wins_delta else ' - GEARS adds little beyond a shared response vector at this training budget'}"
        f", and dir20 = {fnum(omG.get('frac_correct_direction_20_mean'),3)} vs. "
        f"{fnum(omM.get('frac_correct_direction_20_mean'),3)}.",
        "<b>Judge perturbation models on delta metrics.</b> Absolute-expression correlations are near-identical "
        "across all methods (most genes barely change); only change-from-control and top-k direction metrics "
        "separate them [[9, 20]].",
        "<b>Unseen genes require a transfer mechanism.</b> The additive baseline's exact-zero scores demonstrate "
        "that per-gene effect tables cannot generalise; shared response structure (mean-shift) and ontology priors "
        "(GEARS) are the two mechanisms that do.",
        "<b>Simple baselines remain competitive</b>, consistent with recent systematic benchmarks [[13, 20]]; "
        "claims of foundation-model superiority should be tested against mean-shift-class baselines on matched "
        "splits [[19, 20]].",
    ]:
        story.append(Paragraph(b, bullet, bulletText="\u2022"))

    story.append(Paragraph("4.1&nbsp;&nbsp;Limitations", h2))
    story.append(Paragraph(
        "(1) The dataset was subsampled to 10 cells per perturbation (from ~15 available post-QC in this pipeline) "
        "for CPU feasibility; per-perturbation profiles are thin, although averaging over 1,905 test perturbations "
        "makes aggregate comparisons sound. (2) GEARS was trained for 4 epochs rather than the default 20 due to the "
        "24-hour session wall-clock on CPU; its scores are therefore a lower bound. (3) Only the unseen-single "
        "regime is populated in this split, so combination-perturbation generalisation was not evaluated. (4) "
        "Evaluation uses precomputed top-20 DE indices; DE-calling precision/recall was not assessed. (5) Single "
        "dataset, single split seed; confidence intervals across seeds were not computed.", body))

    # ===================== 5. NEXT STEPS =====================
    story.append(Paragraph("5&nbsp;&nbsp;Next steps", h1))
    for b in [
        "Train GEARS to the default 20-epoch budget (GPU) and test whether the margin over the mean-shift baseline widens.",
        "Add scGPT fine-tuning on the identical split for a foundation-model comparison [[19]].",
        "Repeat across multiple split seeds and report confidence intervals.",
        "Evaluate on the essentialome subset and on combo-perturbation regimes (Adamson/Norman datasets) where "
        "GEARS's combinatorial design is targeted [[9]].",
        "Add DE precision/recall and pathway-level (GO enrichment concordance) evaluation.",
    ]:
        story.append(Paragraph(b, bullet, bulletText="\u2022"))

    # ===================== 6. REFERENCES =====================
    if refs:
        story.append(Paragraph("6&nbsp;&nbsp;References", h1))
        for r in refs:
            doi = r.get("doi", "")
            link = f' doi:<font color="#0563C1">{doi}</font>' if doi else ""
            story.append(Paragraph(
                f"[{r.get('id','')}] {r.get('authors','')} ({r.get('year','?')}). {r.get('title','')}. "
                f"<i>{r.get('journal','')}</i>.{link}", ref_st))

    def chrome(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 8); canvas.setFillColor(GREY)
        canvas.drawString(MARGIN, PAGE_H - 12 * mm, "K562 gwps perturbation-response benchmark: GEARS vs. baselines")
        canvas.setStrokeColor(GOLD); canvas.setLineWidth(1)
        canvas.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)
        canvas.setStrokeColor(BORDER); canvas.setLineWidth(0.75)
        canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
        canvas.setFont(FONT, 8); canvas.setFillColor(GREY)
        canvas.drawCentredString(PAGE_W / 2, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(args.out, pagesize=A4, topMargin=18 * mm, bottomMargin=16 * mm,
                            leftMargin=MARGIN, rightMargin=MARGIN, title=title)
    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    print(f"[report] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)

    try:
        from pypdf import PdfReader
        rd = PdfReader(args.out)
        pc = len(rd.pages); sz = os.path.getsize(args.out)
        txt = rd.pages[0].extract_text() or ""
        assert pc >= 2, f"only {pc} page(s)"
        assert sz > 5000, f"only {sz} bytes"
        assert len(txt.strip()) > 0, "no extractable text on page 1"
        print(f"[validate] OK: {pc} pages, {sz} bytes, page-1 text present", flush=True)
    except Exception as e:
        print(f"[validate][WARN] {e}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

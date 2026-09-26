# Method figure

**Paper figure: `fig_pipeline_compact.pdf`** — 5.5 × 2.55 in at `\linewidth`,
**28 % of a 9 in NeurIPS page**. Use the PDF in LaTeX (single-page vector, embedded
fonts, scales losslessly). `.png` is 4200 × 1944 px (764 DPI at 5.5 in) for slides or
Word. `.svg` is the editable source.

`fig_pipeline.*` is the same diagram expanded to a full page — for slides or an
appendix only; it is 85 % of a page.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_pipeline_compact.pdf}
  \caption{...}
  \label{fig:pipeline}
\end{figure}
```

Smallest type is 8.5 units = **4.8 pt** at 5.5 in (the `4a` / `4b` badges only); body
text is 10–12 units = **5.7–6.8 pt**. Do not scale below `\linewidth`.

## Caption

> **Figure 1: One audit path for nine feature sources.** Each source is reduced to a
> single per-note matrix (2) before any evaluation; the audit (3) selects one feature
> per ICD-9 code on shards 0–30 and scores it on held-out shards 281–311. Selected
> features are then evaluated two ways: a forced-choice concordance test (4a) and a
> causal ablation test (4b).

## Scope

The figure states **no results and no comparative claims** — only what was run. The
only quantities in it are protocol facts (corpus size, layer, `d_model`, the 46-code
panel, the shard split, the 9-option slate, the BH-FDR level).

Deliberately not in the figure, because each invites a question the figure cannot
answer and the text handles better: dictionary sizes *k*; which sources see the labels
beyond the group title; the zero- vs mean-ablation and directional-ablation variants;
chance rate for the 9-option slate; judge identities and the $|r|$ bands.

## Provenance

Read from `results/RESULTS.md` — §1 (corpus, 46-code panel, splits), §13.1 (the nine
sources), §13 (the shared audit), §14 (concordance protocol), §11 and §16 (ablation).

## Regenerating

Edit the `.svg`, then (macOS; the SVGs use stock Avenir Next + Menlo):

```bash
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
D=$(pwd)/docs/figures; F=fig_pipeline_compact; W=700; H=324

printf '<style>html,body{margin:0;background:#fff}img{display:block;width:%dpx;height:%dpx}</style><img src="file://%s/%s.svg">' $W $H "$D" $F > /tmp/r.html
"$CH" --headless --disable-gpu --screenshot="$D/$F.png" --window-size=$W,$H \
      --force-device-scale-factor=6 --hide-scrollbars file:///tmp/r.html

printf '<style>@page{size:%sin %sin;margin:0}html,body{margin:0;background:#fff}img{display:block;width:%dpx;height:%dpx}</style><img src="file://%s/%s.svg">' \
  $(python3 -c "print($W/96)") $(python3 -c "print($H/96)") $W $H "$D" $F > /tmp/rp.html
"$CH" --headless --disable-gpu --print-to-pdf="$D/$F.pdf" --no-pdf-header-footer file:///tmp/rp.html
```

For `fig_pipeline` use `W=1220 H=1700`.

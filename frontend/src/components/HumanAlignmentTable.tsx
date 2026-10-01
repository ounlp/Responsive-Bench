import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import {
  fetchAlignmentBenchmark,
  fetchAlignmentStress,
  fetchHumanAlignment,
  fetchInterAnnotatorBenchmark,
  fetchInterAnnotatorCombined,
  fetchInterAnnotatorStress,
  type DimensionAlignment,
  type HumanAlignmentData,
  type InterAnnotatorData,
  type InterAnnotatorDimension,
} from "../data";

// Correlation ≥ threshold is highlighted (follows Liu et al. 2026 convention)
const CORR_HIGHLIGHT_THRESHOLD = 0.25;

function fmt(v: number | null, digits = 3): string {
  if (v === null || v === undefined) return "—";
  return v.toFixed(digits);
}

function fmtPct(v: number | null): string {
  if (v === null || v === undefined) return "—";
  return (v * 100).toFixed(1) + "%";
}

interface CellProps {
  value: number | null;
  isCorrelation?: boolean;
  format?: (v: number | null) => string;
}

function MetricCell({ value, isCorrelation = false, format = fmt }: CellProps) {
  const highlight =
    isCorrelation && value !== null && value >= CORR_HIGHLIGHT_THRESHOLD;
  return (
    <TableCell
      align="right"
      sx={{
        fontVariantNumeric: "tabular-nums",
        fontWeight: highlight ? 700 : 400,
        color: highlight ? "success.dark" : "text.primary",
        whiteSpace: "nowrap",
      }}
    >
      {format(value)}
    </TableCell>
  );
}

/** A correlation cell with its 95% CI (webpage-level clustered bootstrap)
 * shown as a smaller line underneath, when available. */
function CorrelationCell({
  value,
  ciLow,
  ciHigh,
}: {
  value: number | null;
  ciLow: number | null;
  ciHigh: number | null;
}) {
  const highlight = value !== null && value >= CORR_HIGHLIGHT_THRESHOLD;
  return (
    <TableCell
      align="right"
      sx={{
        fontVariantNumeric: "tabular-nums",
        fontWeight: highlight ? 700 : 400,
        color: highlight ? "success.dark" : "text.primary",
        whiteSpace: "nowrap",
      }}
    >
      {fmt(value)}
      {ciLow !== null && ciHigh !== null && (
        <Typography
          component="div"
          variant="caption"
          color="text.secondary"
          sx={{ fontVariantNumeric: "tabular-nums", lineHeight: 1.2, fontWeight: 400 }}
        >
          [{fmt(ciLow)}, {fmt(ciHigh)}]
        </Typography>
      )}
    </TableCell>
  );
}

interface RowProps {
  row: DimensionAlignment | (Omit<DimensionAlignment, "key"> & { label: string });
  isOverall?: boolean;
}

function AlignmentRow({ row, isOverall = false }: RowProps) {
  return (
    <TableRow
      sx={{
        backgroundColor: isOverall ? "action.hover" : undefined,
        "& td": isOverall ? { fontWeight: 600 } : {},
      }}
    >
      <TableCell sx={{ whiteSpace: "nowrap" }}>{row.label}</TableCell>
      <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
        {row.n}
      </TableCell>
      <MetricCell value={row.mse} />
      <MetricCell value={row.mae} />
      <MetricCell value={row.acc} format={fmtPct} />
      <MetricCell value={row.acc1} format={fmtPct} />
      <CorrelationCell value={row.pearson} ciLow={row.pearson_ci_low} ciHigh={row.pearson_ci_high} />
      <CorrelationCell value={row.spearman} ciLow={row.spearman_ci_low} ciHigh={row.spearman_ci_high} />
      <MetricCell value={row.kendall} isCorrelation />
    </TableRow>
  );
}

export interface SubsetSummaryRow {
  subset: string;
  webpages: number;
  overall: HumanAlignmentData["overall"];
}

/** One-row-per-subset overview (Responsive-Bench / Stress test / Combined),
 * each row being that subset's pooled "overall" alignment stats. Pearson
 * shows its webpage-level clustered-bootstrap 95% CI underneath. */
export function SubsetSummaryTable({ rows }: { rows: SubsetSummaryRow[] }) {
  return (
    <TableContainer component={Paper} variant="outlined" sx={{ maxWidth: 1000, mb: 3 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Subset</TableCell>
            <TableCell align="right">Webpages</TableCell>
            <Tooltip title="Pooled (webpage, dimension) rating pairs" placement="top">
              <TableCell align="right">Rating pairs</TableCell>
            </Tooltip>
            <Tooltip title="Pearson correlation, with 95% CI from a webpage-level clustered bootstrap (2000 resamples)" placement="top">
              <TableCell align="right">Pearson↑</TableCell>
            </Tooltip>
            <Tooltip title="Spearman rank correlation (higher is better)" placement="top">
              <TableCell align="right">Spearman↑</TableCell>
            </Tooltip>
            <Tooltip title="Mean absolute error (lower is better)" placement="top">
              <TableCell align="right">MAE↓</TableCell>
            </Tooltip>
            <Tooltip title="Within-1 accuracy: |human − judge| ≤ 1 (higher is better)" placement="top">
              <TableCell align="right">ACC±1↑</TableCell>
            </Tooltip>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => {
            const isCombined = row.subset === "Combined";
            return (
              <TableRow
                key={row.subset}
                sx={{
                  backgroundColor: isCombined ? "action.hover" : undefined,
                  "& td": isCombined ? { fontWeight: 600 } : {},
                }}
              >
                <TableCell sx={{ whiteSpace: "nowrap" }}>{row.subset}</TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                  {row.webpages}
                </TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                  {row.overall.n}
                </TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                  {fmt(row.overall.pearson)}
                  {row.overall.pearson_ci_low !== null && row.overall.pearson_ci_high !== null && (
                    <Typography
                      component="div"
                      variant="caption"
                      color="text.secondary"
                      sx={{ fontVariantNumeric: "tabular-nums", lineHeight: 1.2 }}
                    >
                      [{fmt(row.overall.pearson_ci_low)}, {fmt(row.overall.pearson_ci_high)}]
                    </Typography>
                  )}
                </TableCell>
                <MetricCell value={row.overall.spearman} isCorrelation />
                <MetricCell value={row.overall.mae} />
                <MetricCell value={row.overall.acc1} format={fmtPct} />
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

/** Pure rendering of one alignment result set — reused by the default RunsList
 * section below and by the AlignmentComparison page, which fetches whichever
 * variant the user picks from the judge-model dropdown. */
export function AlignmentResultsTable({
  data,
  heading = "Human Annotation vs LLM Judge Alignment",
}: {
  data: HumanAlignmentData;
  heading?: string;
}) {
  const generatedDate = new Date(data.generated_at).toLocaleDateString(
    undefined,
    { year: "numeric", month: "short", day: "numeric" },
  );

  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom sx={{ mt: 4 }}>
        {heading}
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Alignment between human Likert scores and LLM judge scores (1–5 scale)
        across {data.n_matched_samples} matched samples from{" "}
        {data.n_annotations} annotations. Correlations ≥{" "}
        {CORR_HIGHLIGHT_THRESHOLD} are{" "}
        <Box
          component="span"
          sx={{ fontWeight: 700, color: "success.dark" }}
        >
          highlighted
        </Box>
        . Generated {generatedDate}.
      </Typography>

      <TableContainer
        component={Paper}
        variant="outlined"
        sx={{ maxWidth: 1000 }}
      >
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Dimension</TableCell>
              <TableCell align="right">n</TableCell>
              <Tooltip title="Mean squared error (lower is better)" placement="top">
                <TableCell align="right">MSE↓</TableCell>
              </Tooltip>
              <Tooltip title="Mean absolute error (lower is better)" placement="top">
                <TableCell align="right">MAE↓</TableCell>
              </Tooltip>
              <Tooltip title="Exact match accuracy (higher is better)" placement="top">
                <TableCell align="right">ACC↑</TableCell>
              </Tooltip>
              <Tooltip title="Within-1 accuracy: |human − judge| ≤ 1 (higher is better)" placement="top">
                <TableCell align="right">ACC±1↑</TableCell>
              </Tooltip>
              <Tooltip title="Pearson correlation (higher is better)" placement="top">
                <TableCell align="right">Pearson↑</TableCell>
              </Tooltip>
              <Tooltip title="Spearman rank correlation (higher is better)" placement="top">
                <TableCell align="right">Spearman↑</TableCell>
              </Tooltip>
              <Tooltip title="Kendall's τ-b (higher is better)" placement="top">
                <TableCell align="right">Kendall↑</TableCell>
              </Tooltip>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.dimensions.map((dim) => (
              <AlignmentRow key={dim.key} row={dim} />
            ))}
            <AlignmentRow row={data.overall} isOverall />
          </TableBody>
        </Table>
      </TableContainer>
    </section>
  );
}

interface InterAnnotatorRowProps {
  row: InterAnnotatorDimension | Omit<InterAnnotatorDimension, "key">;
  isOverall?: boolean;
}

function InterAnnotatorRow({ row, isOverall = false }: InterAnnotatorRowProps) {
  return (
    <TableRow
      sx={{
        backgroundColor: isOverall ? "action.hover" : undefined,
        "& td": isOverall ? { fontWeight: 600 } : {},
      }}
    >
      <TableCell sx={{ whiteSpace: "nowrap" }}>{row.label}</TableCell>
      <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
        {row.n}
      </TableCell>
      <MetricCell value={row.mse} />
      <MetricCell value={row.mae} />
      <MetricCell value={row.acc} format={fmtPct} />
      <MetricCell value={row.acc1} format={fmtPct} />
      <CorrelationCell value={row.pearson} ciLow={row.pearson_ci_low} ciHigh={row.pearson_ci_high} />
      <CorrelationCell value={row.spearman} ciLow={row.spearman_ci_low} ciHigh={row.spearman_ci_high} />
      <MetricCell value={row.kendall} isCorrelation />
    </TableRow>
  );
}

/** Pure rendering of one inter-annotator (human--human) result set. Each
 * dimension's values are already the average across every annotator pair
 * (see run_inter_annotator in export_human_alignment.py). */
export function InterAnnotatorResultsTable({
  data,
  heading = "Inter-Annotator Agreement",
}: {
  data: InterAnnotatorData;
  heading?: string;
}) {
  const pairLabel = data.annotator_pairs.map(([a, b]) => `${a}–${b}`).join(", ");
  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom sx={{ mt: 4 }}>
        {heading}
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Agreement between the {data.annotator_ids.length} human annotators
        themselves, averaged across the {data.annotator_pairs.length}{" "}
        pairwise comparisons ({pairLabel}) on their jointly-rated webpages,
        across {data.n_webpages} webpages.
      </Typography>

      <TableContainer component={Paper} variant="outlined" sx={{ maxWidth: 1000 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Dimension</TableCell>
              <TableCell align="right">n</TableCell>
              <Tooltip title="Mean squared error (lower is better)" placement="top">
                <TableCell align="right">MSE↓</TableCell>
              </Tooltip>
              <Tooltip title="Mean absolute error (lower is better)" placement="top">
                <TableCell align="right">MAE↓</TableCell>
              </Tooltip>
              <Tooltip title="Exact match accuracy (higher is better)" placement="top">
                <TableCell align="right">ACC↑</TableCell>
              </Tooltip>
              <Tooltip title="Within-1 accuracy: |annotator A − annotator B| ≤ 1 (higher is better)" placement="top">
                <TableCell align="right">ACC±1↑</TableCell>
              </Tooltip>
              <Tooltip title="Pearson correlation (higher is better)" placement="top">
                <TableCell align="right">Pearson↑</TableCell>
              </Tooltip>
              <Tooltip title="Spearman rank correlation (higher is better)" placement="top">
                <TableCell align="right">Spearman↑</TableCell>
              </Tooltip>
              <Tooltip title="Kendall's τ-b (higher is better)" placement="top">
                <TableCell align="right">Kendall↑</TableCell>
              </Tooltip>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.dimensions.map((dim) => (
              <InterAnnotatorRow key={dim.key} row={dim} />
            ))}
            <InterAnnotatorRow row={data.overall} isOverall />
          </TableBody>
        </Table>
      </TableContainer>
    </section>
  );
}

export interface InterAnnotatorSummaryRow {
  subset: string;
  webpages: number;
  overall: InterAnnotatorData["overall"];
}

/** One-row-per-subset overview, mirroring SubsetSummaryTable above but for
 * human--human agreement (no Pearson CI -- combining a CI across three
 * annotator pairs isn't a single well-defined statistic). */
export function InterAnnotatorSummaryTable({ rows }: { rows: InterAnnotatorSummaryRow[] }) {
  return (
    <TableContainer component={Paper} variant="outlined" sx={{ maxWidth: 1000, mb: 3 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Subset</TableCell>
            <TableCell align="right">Webpages</TableCell>
            <Tooltip title="Pooled (webpage, dimension) rating pairs, per annotator pair" placement="top">
              <TableCell align="right">Rating pairs</TableCell>
            </Tooltip>
            <Tooltip title="Pearson correlation, averaged across the three annotator pairs" placement="top">
              <TableCell align="right">Pearson↑</TableCell>
            </Tooltip>
            <Tooltip title="Spearman rank correlation (higher is better)" placement="top">
              <TableCell align="right">Spearman↑</TableCell>
            </Tooltip>
            <Tooltip title="Mean absolute error (lower is better)" placement="top">
              <TableCell align="right">MAE↓</TableCell>
            </Tooltip>
            <Tooltip title="Within-1 accuracy: |annotator A − annotator B| ≤ 1 (higher is better)" placement="top">
              <TableCell align="right">ACC±1↑</TableCell>
            </Tooltip>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => {
            const isCombined = row.subset === "Combined";
            return (
              <TableRow
                key={row.subset}
                sx={{
                  backgroundColor: isCombined ? "action.hover" : undefined,
                  "& td": isCombined ? { fontWeight: 600 } : {},
                }}
              >
                <TableCell sx={{ whiteSpace: "nowrap" }}>{row.subset}</TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                  {row.webpages}
                </TableCell>
                <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                  {row.overall.n}
                </TableCell>
                <MetricCell value={row.overall.pearson} isCorrelation />
                <MetricCell value={row.overall.spearman} isCorrelation />
                <MetricCell value={row.overall.mae} />
                <MetricCell value={row.overall.acc1} format={fmtPct} />
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

interface ThreeWayAlignment {
  benchmark: HumanAlignmentData | null;
  stress: HumanAlignmentData | null;
  combined: HumanAlignmentData | null;
}

interface ThreeWayInterAnnotator {
  benchmark: InterAnnotatorData | null;
  stress: InterAnnotatorData | null;
  combined: InterAnnotatorData | null;
}

export default function HumanAlignmentTable() {
  const [data, setData] = useState<ThreeWayAlignment | null>(null);
  const [interAnnotator, setInterAnnotator] = useState<ThreeWayInterAnnotator | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchAlignmentBenchmark(),
      fetchAlignmentStress(),
      fetchHumanAlignment(),
      fetchInterAnnotatorBenchmark(),
      fetchInterAnnotatorStress(),
      fetchInterAnnotatorCombined(),
    ])
      .then(([benchmark, stress, combined, iaBenchmark, iaStress, iaCombined]) => {
        setData({ benchmark, stress, combined });
        setInterAnnotator({ benchmark: iaBenchmark, stress: iaStress, combined: iaCombined });
        setLoading(false);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <Typography
        component="p"
        sx={{ display: "flex", alignItems: "center", gap: 1, mt: 2 }}
      >
        <CircularProgress size={20} /> Loading alignment data…
      </Typography>
    );
  }

  if (error) {
    return <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>;
  }

  if (!data || (!data.benchmark && !data.stress && !data.combined)) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        No human alignment data available. Run{" "}
        <Typography
          component="span"
          variant="body2"
          sx={{ fontFamily: "monospace" }}
        >
          python scripts/export_human_alignment.py
        </Typography>{" "}
        to generate it.
      </Alert>
    );
  }

  const summaryRows: SubsetSummaryRow[] = [
    data.benchmark && { subset: "Responsive-Bench", webpages: data.benchmark.n_matched_samples, overall: data.benchmark.overall },
    data.stress && { subset: "Stress test", webpages: data.stress.n_matched_samples, overall: data.stress.overall },
    data.combined && { subset: "Combined", webpages: data.combined.n_matched_samples, overall: data.combined.overall },
  ].filter((r): r is SubsetSummaryRow => Boolean(r));

  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom sx={{ mt: 4 }}>
        Human Annotation vs LLM Judge Alignment
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Overview across the two human-annotated subsets that make up the full
        validation set: 30 pages sampled from the released Responsive-Bench
        benchmark (full 6-dimension rubric), and 26 curated stress-test pages
        (sketch-independent 5-dimension rubric, no Sketch Fidelity). Pearson's
        95% CI is a webpage-level clustered bootstrap.
      </Typography>
      <SubsetSummaryTable rows={summaryRows} />

      {data.benchmark && (
        <AlignmentResultsTable data={data.benchmark} heading="Responsive-Bench Subset (30 webpages)" />
      )}
      {data.stress && (
        <AlignmentResultsTable data={data.stress} heading="Stress-Test Subset (26 webpages)" />
      )}
      {data.combined && (
        <AlignmentResultsTable data={data.combined} heading="Combined (56 webpages)" />
      )}

      {interAnnotator && (interAnnotator.benchmark || interAnnotator.stress || interAnnotator.combined) && (
        <>
          <Typography variant="h5" component="h2" gutterBottom sx={{ mt: 5 }}>
            Inter-Annotator Agreement
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Agreement among the human annotators themselves, split the same
            three ways as above, to check whether low human--LLM agreement on
            a subset reflects a judge weakness or is simply how much humans
            agree with each other on that subset.
          </Typography>
          <InterAnnotatorSummaryTable
            rows={[
              interAnnotator.benchmark && {
                subset: "Responsive-Bench",
                webpages: interAnnotator.benchmark.n_webpages,
                overall: interAnnotator.benchmark.overall,
              },
              interAnnotator.stress && {
                subset: "Stress test",
                webpages: interAnnotator.stress.n_webpages,
                overall: interAnnotator.stress.overall,
              },
              interAnnotator.combined && {
                subset: "Combined",
                webpages: interAnnotator.combined.n_webpages,
                overall: interAnnotator.combined.overall,
              },
            ].filter((r): r is InterAnnotatorSummaryRow => Boolean(r))}
          />

          {interAnnotator.benchmark && (
            <InterAnnotatorResultsTable
              data={interAnnotator.benchmark}
              heading="Responsive-Bench Subset (30 webpages)"
            />
          )}
          {interAnnotator.stress && (
            <InterAnnotatorResultsTable
              data={interAnnotator.stress}
              heading="Stress-Test Subset (26 webpages)"
            />
          )}
          {interAnnotator.combined && (
            <InterAnnotatorResultsTable data={interAnnotator.combined} heading="Combined (56 webpages)" />
          )}
        </>
      )}
    </section>
  );
}

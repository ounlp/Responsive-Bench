import { useEffect, useState } from "react";
import { Link as RouterLink, useParams } from "react-router-dom";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import { fetchJson, assetUrl, type EvalSamplesResponse, type SampleMetricsResponse } from "../data";
import ImageWithFallback from "../ImageWithFallback";

const METRIC_LABELS: Record<string, string> = {
  layout_similarity: "Layout similarity",
  horizontal_scroll: "Horizontal scroll",
  llm_judge_screenshots: "LLM judge",
  tap_targets: "Tap targets",
  line_spacing: "Line spacing",
  viewport_meta_tag: "Viewport meta tag",
  responsive_media: "Responsive media",
  relative_units: "Relative units",
  computed_font_sizes: "Computed font sizes",
  rotation_friendly: "Rotation friendly",
  text_media_cutoff_overlap: "Text/media cutoff & overlap",
  heading_order: "Heading order",
  text_enlargement: "Text enlargement",
  accessibility: "Accessibility (axe-core)",
  accessible_landmarks: "Accessible landmarks",
  accessible_hamburger_menu: "Accessible hamburger menu",
  accessible_active_nav_item: "Accessible active nav item",
  accessible_form_labels: "Accessible form labels",
  accessible_no_keyboard_trap: "No keyboard trap",
};

const DIMENSION_LABELS: Record<string, string> = {
  fluid_layout_and_container_adaptation: "Fluid layout & container adaptation",
  flexible_grid_reflow: "Flexible grid reflow",
  responsive_media: "Responsive media",
  intentional_content_visibility: "Intentional content visibility",
  proper_alignment_and_spacing: "Proper alignment and spacing",
  sketch_fidelity: "Sketch fidelity",
  // Legacy keys (older eval / annotation JSON)
  fluid_foundations: "Fluid foundations",
  adaptive_containers: "Adaptive containers",
};

function formatViewportFractions(payload: Record<string, unknown>): string {
  const vr = payload.viewport_results as Record<string, Record<string, unknown>> | undefined;
  if (!vr) return "—";
  return Object.entries(vr)
    .map(([vp, r]) => {
      if (r == null || typeof r !== "object") return null;
      const passed = r.passed as number | undefined;
      const checked = r.checked as number | undefined;
      const score = r.score as number | null | undefined;
      if (typeof passed === "number" && typeof checked === "number") {
        const pct = checked > 0 ? Math.round((passed / checked) * 100) : 0;
        return `${vp}: ${passed}/${checked} (${pct}%)`;
      }
      if (score == null) return `${vp}: N/A`;
      return `${vp}: ${score}/5`;
    })
    .filter(Boolean)
    .join(" · ");
}

function formatMetric(name: string, payload: Record<string, unknown>): { label: string; summary: string } {
  const label = METRIC_LABELS[name] ?? name;

  if (name === "llm_judge_screenshots") {
    const overall = payload.score_overall as number | undefined;
    return { label, summary: overall != null ? `Overall score: ${overall}` : "" };
  }

  if (name === "horizontal_scroll") {
    const needed = payload.horizontal_scroll_needed as boolean | undefined;
    return { label, summary: needed === true ? "Failed" : "Passed" };
  }

  if (name === "layout_similarity") {
    const iou = payload.weighted_iou as number | undefined;
    return { label, summary: iou != null ? `Weighted IoU: ${iou.toFixed(3)}` : "—" };
  }

  // LLM-judged single-score metrics
  if (
    name === "accessible_landmarks" ||
    name === "accessible_active_nav_item" ||
    name === "accessible_hamburger_menu"
  ) {
    if (name === "accessible_hamburger_menu" && payload.reason === "not_applicable") {
      return { label, summary: "Not applicable" };
    }
    const score = payload.score as number | null | undefined;
    return { label, summary: score != null ? `Score: ${score}/5` : "—" };
  }

  // Viewport-based fraction metrics
  if (name === "accessible_form_labels" || name === "accessible_no_keyboard_trap") {
    return { label, summary: formatViewportFractions(payload) };
  }

  if (name === "accessibility") {
    const passed = payload.passed as boolean | undefined;
    if (passed === true) return { label, summary: "Passed" };
    const totalViolationsCount = payload.total_violations_count as number | undefined;
    return { label, summary: totalViolationsCount != null ? `${totalViolationsCount} violation(s)` : "Failed" };
  }

  const passed = payload.passed as boolean | undefined;
  if (passed === true) return { label, summary: "Passed" };

  const violationsCount = (payload.violations_count as number) ?? (payload.violation_count as number);
  if (typeof violationsCount === "number") {
    return { label, summary: `${violationsCount} violation(s)` };
  }

  const totalViolations = payload.total_violations as number | undefined;
  if (typeof totalViolations === "number") {
    return { label, summary: `${totalViolations} violation(s)` };
  }

  return { label, summary: payload.status === "success" ? "—" : "Error" };
}

export default function EvalResults() {
  const { runId, evalId, sampleId } = useParams<{ runId: string; evalId: string; sampleId: string }>();
  const [data, setData] = useState<SampleMetricsResponse | null>(null);
  const [samplesData, setSamplesData] = useState<EvalSamplesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId || !evalId || !sampleId) return;
    fetchJson<SampleMetricsResponse>(`runs/${runId}/evals/${evalId}/samples/${sampleId}.json`)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [runId, evalId, sampleId]);

  useEffect(() => {
    if (!runId || !evalId) return;
    fetchJson<EvalSamplesResponse>(`runs/${runId}/evals/${evalId}.json`)
      .then(setSamplesData)
      .catch(() => {
        // Non-fatal: navigation buttons will fall back to numeric increment/decrement when possible.
        setSamplesData(null);
      });
  }, [runId, evalId]);

  if (error) return <Typography color="error">{error}</Typography>;
  if (!runId || !evalId || !sampleId || !data) return <Typography>Loading…</Typography>;

  const metrics = data.metrics ?? {};
  const renderDesktop = data.render_desktop_path ?? null;
  const renderMobile = data.render_mobile_path ?? null;
  const renderTablet = data.render_tablet_path ?? null;
  const referencePath = data.reference_path ?? null;
  const sketchDesktopPath = data.sketch_desktop_path ?? null;
  const sketchMobilePath = data.sketch_mobile_path ?? null;
  const hasSketchesOrReference = referencePath || sketchDesktopPath || sketchMobilePath;

  const llmJudge = metrics.llm_judge_screenshots as Record<string, unknown> | undefined;
  const llmScores = llmJudge?.scores as Record<string, number> | undefined;
  const llmRationale = llmJudge?.rationale as Record<string, string> | undefined;

  const sampleIds = samplesData?.samples?.map((s) => s.id) ?? null;
  const currentIdx = sampleIds ? sampleIds.indexOf(sampleId) : -1;
  const prevSampleId = sampleIds && currentIdx > 0 ? sampleIds[currentIdx - 1] : null;
  const nextSampleId =
    sampleIds && currentIdx >= 0 && currentIdx < sampleIds.length - 1 ? sampleIds[currentIdx + 1] : null;

  const SampleNav = ({ sx }: { sx?: Record<string, unknown> }) => (
    <Stack direction="row" spacing={1} justifyContent="space-between" alignItems="center" sx={sx}>
      <Button
        component={RouterLink}
        variant="outlined"
        size="small"
        to={
          prevSampleId
            ? `/runs/${runId}/evals/${evalId}/samples/${prevSampleId}/results`
            : `/runs/${runId}/evals/${evalId}/samples/${sampleId}/results`
        }
        disabled={!prevSampleId}
      >
        ← Previous
      </Button>
      <Typography variant="body2" color="text.secondary">
        Sample {sampleId}
        {sampleIds && currentIdx >= 0 ? ` (${currentIdx + 1}/${sampleIds.length})` : ""}
      </Typography>
      <Button
        component={RouterLink}
        variant="outlined"
        size="small"
        to={
          nextSampleId
            ? `/runs/${runId}/evals/${evalId}/samples/${nextSampleId}/results`
            : `/runs/${runId}/evals/${evalId}/samples/${sampleId}/results`
        }
        disabled={!nextSampleId}
      >
        Next →
      </Button>
    </Stack>
  );

  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom>
        Eval Results — Sample {sampleId}
      </Typography>
      <Typography paragraph>
        <Link component={RouterLink} to={`/runs/${runId}/evals/${evalId}`} underline="hover">
          ← Back to {evalId}
        </Link>
      </Typography>

      <SampleNav sx={{ mb: 2 }} />

      {hasSketchesOrReference && (
        <>
          <Typography variant="h6" component="h3" gutterBottom>
            Sketches & reference
          </Typography>
          <Stack direction="row" flexWrap="wrap" useFlexGap spacing={2} sx={{ mb: 2 }}>
            {sketchDesktopPath && (
              <Box sx={{ flex: "1 1 200px", minWidth: 200 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Sketch (Desktop)
                </Typography>
                <ImageWithFallback
                  src={assetUrl(sketchDesktopPath)}
                  alt="Sketch Desktop"
                  style={{ width: "100%", height: "auto" }}
                />
              </Box>
            )}
            {sketchMobilePath && (
              <Box sx={{ flex: "1 1 200px", minWidth: 200 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Sketch (Mobile)
                </Typography>
                <ImageWithFallback
                  src={assetUrl(sketchMobilePath)}
                  alt="Sketch Mobile"
                  style={{ width: "100%", height: "auto" }}
                />
              </Box>
            )}
            {referencePath && (
              <Box sx={{ flex: "1 1 200px", minWidth: 200 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Reference
                </Typography>
                <ImageWithFallback
                  src={assetUrl(referencePath)}
                  alt="Reference"
                  style={{ width: "100%", height: "auto" }}
                />
              </Box>
            )}
          </Stack>
        </>
      )}

      <Typography variant="h6" component="h3" gutterBottom>
        Rendered viewports
      </Typography>
      <Stack direction="row" flexWrap="wrap" useFlexGap spacing={2} sx={{ mb: 3 }}>
        <Box sx={{ flex: "1 1 200px", minWidth: 200 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Desktop
          </Typography>
          <ImageWithFallback
            src={assetUrl(renderDesktop)}
            alt="Desktop"
            style={{ width: "100%", height: "auto" }}
          />
        </Box>
        <Box sx={{ flex: "1 1 200px", minWidth: 200 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Mobile
          </Typography>
          <ImageWithFallback
            src={assetUrl(renderMobile)}
            alt="Mobile"
            style={{ width: "100%", height: "auto" }}
          />
        </Box>
        <Box sx={{ flex: "1 1 200px", minWidth: 200 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Tablet
          </Typography>
          <ImageWithFallback
            src={assetUrl(renderTablet)}
            alt="Tablet"
            style={{ width: "100%", height: "auto" }}
          />
        </Box>
      </Stack>

      <Typography variant="h6" component="h3" gutterBottom>
        Metrics
      </Typography>
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow sx={{ bgcolor: "action.hover" }}>
              <TableCell>Metric</TableCell>
              <TableCell>Result</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(metrics).flatMap(([key, value]) => {
              if (!value || typeof value !== "object") return [];
              const v = value as Record<string, unknown>;

              // llm_judge_screenshots: expand each dimension as its own row
              if (key === "llm_judge_screenshots") {
                if (llmScores && llmRationale) {
                  return Object.entries(llmScores).map(([dim, score]) => (
                    <TableRow key={`llm-${dim}`}>
                      <TableCell sx={{ verticalAlign: "top" }}>
                        {DIMENSION_LABELS[dim] ?? dim}
                      </TableCell>
                      <TableCell>
                        <Typography fontWeight="medium">Score: {score}</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                          {llmRationale[dim] ?? "—"}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ));
                }
                const { label, summary } = formatMetric(key, v);
                return [<TableRow key={key}><TableCell>{label}</TableCell><TableCell>{summary}</TableCell></TableRow>];
              }

              // accessibility: expand per-viewport violation rows
              if (key === "accessibility") {
                type Violation = { id: string; impact?: string; description?: string };
                type VpData = { violations?: Violation[]; violations_count?: number };
                const viewports = v.viewports as Record<string, VpData> | undefined;
                const totalCount = v.total_violations_count as number | undefined;
                const isPassed = v.passed as boolean | undefined;

                const impactColor = (impact?: string) => {
                  if (impact === "critical" || impact === "serious") return "error.main";
                  if (impact === "moderate") return "warning.main";
                  return "text.secondary";
                };

                const headerRow = (
                  <TableRow key="accessibility-total">
                    <TableCell sx={{ verticalAlign: "top" }}>Accessibility (axe-core)</TableCell>
                    <TableCell>
                      <Typography fontWeight="medium">
                        {isPassed ? "Passed" : totalCount != null ? `${totalCount} violation(s) across viewports` : "Failed"}
                      </Typography>
                    </TableCell>
                  </TableRow>
                );

                if (!viewports || isPassed) return [headerRow];

                const vpRows = Object.entries(viewports).flatMap(([vpName, vpData]) => {
                  const violations = vpData.violations ?? [];
                  if (violations.length === 0) return [];
                  return [
                    <TableRow key={`accessibility-${vpName}`}>
                      <TableCell sx={{ verticalAlign: "top", color: "text.secondary", pl: 4 }}>
                        {vpName} ({vpData.violations_count ?? violations.length})
                      </TableCell>
                      <TableCell>
                        <Stack spacing={0.5}>
                          {violations.map((viol, i) => (
                            <Box key={i}>
                              <Typography variant="body2">
                                <Box component="span" sx={{ fontWeight: "medium" }}>{viol.id}</Box>
                                {viol.impact && (
                                  <Box component="span" sx={{ color: impactColor(viol.impact), ml: 0.5 }}>
                                    [{viol.impact}]
                                  </Box>
                                )}
                              </Typography>
                              {viol.description && (
                                <Typography variant="body2" color="text.secondary">{viol.description}</Typography>
                              )}
                            </Box>
                          ))}
                        </Stack>
                      </TableCell>
                    </TableRow>,
                  ];
                });

                return [headerRow, ...vpRows];
              }

              // line_spacing: show violation count + per-violation selector and ratio
              if (key === "line_spacing") {
                const { label, summary } = formatMetric(key, v);
                type LsViolation = { selector?: string; tag?: string; ratio?: number };
                const violations = v.violations as LsViolation[] | undefined;
                const minRatio = v.min_ratio as number | undefined;
                return [
                  <TableRow key={key}>
                    <TableCell sx={{ verticalAlign: "top" }}>{label}</TableCell>
                    <TableCell>
                      <Typography fontWeight="medium">{summary}</Typography>
                      {violations && violations.length > 0 && (
                        <Stack spacing={0.25} sx={{ mt: 0.5 }}>
                          {violations.map((viol, i) => (
                            <Typography key={i} variant="body2" color="text.secondary">
                              {viol.selector ?? viol.tag ?? "?"} — ratio: {viol.ratio}
                              {minRatio != null ? ` (min: ${minRatio})` : ""}
                            </Typography>
                          ))}
                        </Stack>
                      )}
                    </TableCell>
                  </TableRow>,
                ];
              }

              // LLM-judged single-score metrics: show score + rationale
              if (
                key === "accessible_landmarks" ||
                key === "accessible_active_nav_item" ||
                key === "accessible_hamburger_menu"
              ) {
                const { label, summary } = formatMetric(key, v);
                const rationale = v.rationale as string | null | undefined;
                return [
                  <TableRow key={key}>
                    <TableCell sx={{ verticalAlign: "top" }}>{label}</TableCell>
                    <TableCell>
                      <Typography fontWeight="medium">{summary}</Typography>
                      {rationale && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                          {rationale}
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>,
                ];
              }

              const { label, summary } = formatMetric(key, v);
              return [
                <TableRow key={key}>
                  <TableCell>{label}</TableCell>
                  <TableCell>{summary}</TableCell>
                </TableRow>,
              ];
            })}
          </TableBody>
        </Table>
      </TableContainer>

      <SampleNav sx={{ mt: 2 }} />
    </section>
  );
}

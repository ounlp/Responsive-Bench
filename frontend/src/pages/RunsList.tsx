import { useEffect, useMemo, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import { LEADERBOARD_ONLY, fetchJson, type EvalLeaderboardRow, type RunsListResponse } from "../data";
import HumanAlignmentTable from "../components/HumanAlignmentTable";
import ScoreComparisonTable from "../components/ScoreComparisonTable";

function sortedRowsForLeaderboard(rows: EvalLeaderboardRow[]): EvalLeaderboardRow[] {
  return [...rows].sort((a, b) => {
    const aScore = a.llm_judge_avg_normalized;
    const bScore = b.llm_judge_avg_normalized;
    // Rows without an LLM judge score (e.g. layout_similarity-only evals) sort after
    // scored rows rather than being excluded from the table entirely.
    if (aScore == null && bScore != null) return 1;
    if (aScore != null && bScore == null) return -1;
    if (aScore != null && bScore != null && aScore !== bScore) {
      return bScore - aScore;
    }
    const rc = a.run_id.localeCompare(b.run_id);
    if (rc !== 0) return rc;
    return a.eval_id.localeCompare(b.eval_id);
  });
}

export default function RunsList() {
  const [data, setData] = useState<RunsListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<RunsListResponse>("runs.json")
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const orderedRows = useMemo(
    () => (data ? sortedRowsForLeaderboard(data.runs) : []),
    [data],
  );

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }
  if (!data) {
    return (
      <Typography component="p" sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <CircularProgress size={20} /> Loading runs…
      </Typography>
    );
  }

  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom>
        Runs
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        LLM judge leaderboard: mean score per eval on a 0–1 scale (1–5 Likert mapped linearly), averaged
        over samples with successful judge output. Regenerate{" "}
        <Typography component="span" variant="body2" sx={{ fontFamily: "monospace" }}>
          data/runs.json
        </Typography>{" "}
        with{" "}
        <Typography component="span" variant="body2" sx={{ fontFamily: "monospace" }}>
          python scripts/build_runs_data.py
        </Typography>
        .
      </Typography>

      {data.runs.length === 0 ? (
        <Typography color="text.secondary">No runs found.</Typography>
      ) : (
        <TableContainer component={Paper} variant="outlined" sx={{ maxWidth: 1150 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Run / Eval</TableCell>
                <TableCell sx={{ whiteSpace: "nowrap" }}>Generation Model</TableCell>
                <TableCell sx={{ whiteSpace: "nowrap" }}>Judge Model</TableCell>
                <TableCell align="right">Avg LLM judge (0–1)</TableCell>
                <Tooltip
                  title="Mean layout similarity IoU averaged across desktop and mobile viewports per sample. Hover a row to see the desktop/mobile breakdown."
                  placement="top"
                >
                  <TableCell align="right" sx={{ whiteSpace: "nowrap", cursor: "help" }}>
                    Layout Sim. (IoU)↑
                  </TableCell>
                </Tooltip>
                <Tooltip
                  title="Mean axe-core violations + accessible_form_labels fails per sample. Lower is better."
                  placement="top"
                >
                  <TableCell align="right" sx={{ whiteSpace: "nowrap", cursor: "help" }}>
                    A11y Violations↓
                  </TableCell>
                </Tooltip>
                <Tooltip
                  title="Mean normalized LLM judge score [0–1] for accessible landmarks, hamburger menu, active nav item, and keyboard trap. Higher is better."
                  placement="top"
                >
                  <TableCell align="right" sx={{ whiteSpace: "nowrap", cursor: "help" }}>
                    A11y Judge (0–1)↑
                  </TableCell>
                </Tooltip>
              </TableRow>
            </TableHead>
            <TableBody>
              {orderedRows.map((r) => {
                const score = r.llm_judge_avg_normalized;
                return (
                  <TableRow key={`${r.run_id}::${r.eval_id}`} hover>
                    <TableCell>
                      {LEADERBOARD_ONLY ? (
                        <Typography component="span" variant="body2">
                          {r.run_id} / {r.eval_id}
                        </Typography>
                      ) : (
                      <>
                      <Link
                        component={RouterLink}
                        to={`/runs/${r.run_id}`}
                        underline="hover"
                        color="primary"
                      >
                        {r.run_id}
                      </Link>
                      <Typography component="span" variant="body2" color="text.secondary" sx={{ mx: 0.5 }}>
                        /
                      </Typography>
                      <Link
                        component={RouterLink}
                        to={`/runs/${r.run_id}/evals/${r.eval_id}`}
                        underline="hover"
                        color="primary"
                      >
                        {r.eval_id}
                      </Link>
                      </>
                      )}
                    </TableCell>
                    <Tooltip title={r.generation_provider ? `Provider: ${r.generation_provider}` : "No generation config found"}>
                      <TableCell sx={{ whiteSpace: "nowrap", cursor: r.generation_provider ? "help" : undefined }}>
                        {r.generation_model ?? "—"}
                      </TableCell>
                    </Tooltip>
                    <Tooltip title={r.judge_provider ? `Provider: ${r.judge_provider}` : "No judge config found"}>
                      <TableCell sx={{ whiteSpace: "nowrap", cursor: r.judge_provider ? "help" : undefined }}>
                        {r.judge_model ?? "—"}
                      </TableCell>
                    </Tooltip>
                    <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                      {score == null ? "—" : score.toFixed(3)}
                    </TableCell>
                    <Tooltip
                      placement="left"
                      title={
                        r.layout_similarity_avg_desktop_iou == null &&
                        r.layout_similarity_avg_tablet_iou == null &&
                        r.layout_similarity_avg_mobile_iou == null
                          ? "No layout similarity data"
                          : `Mobile: ${r.layout_similarity_avg_mobile_iou?.toFixed(3) ?? "—"}  |  Tablet: ${r.layout_similarity_avg_tablet_iou?.toFixed(3) ?? "—"}  |  Desktop: ${r.layout_similarity_avg_desktop_iou?.toFixed(3) ?? "—"}`
                      }
                    >
                      <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                        {r.layout_similarity_avg_iou == null ? "—" : r.layout_similarity_avg_iou.toFixed(3)}
                      </TableCell>
                    </Tooltip>
                    <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                      {r.avg_accessibility_violations == null ? "—" : r.avg_accessibility_violations.toFixed(1)}
                    </TableCell>
                    <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                      {r.avg_accessible_judge_score == null ? "—" : r.avg_accessible_judge_score.toFixed(3)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <HumanAlignmentTable />
      {!LEADERBOARD_ONLY && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          <Link component={RouterLink} to="/alignment-comparison" underline="hover">
            Compare against other judge models →
          </Link>
        </Typography>
      )}
      <ScoreComparisonTable />
    </section>
  );
}

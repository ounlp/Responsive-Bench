import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Button from "@mui/material/Button";
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
import { fetchHumanAlignment, type DimensionAlignment, type HumanAlignmentData } from "../data";

function fmtMeanStd(mean: number | null, std: number | null): string {
  if (mean === null) return "—";
  const stdStr = std !== null ? ` (${std.toFixed(2)})` : "";
  return mean.toFixed(2) + stdStr;
}

type OverallRow = Omit<DimensionAlignment, "key"> & { label: string };

interface RowProps {
  row: DimensionAlignment | OverallRow;
  isOverall?: boolean;
}

function ComparisonRow({ row, isOverall = false }: RowProps) {
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
      <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
        {fmtMeanStd(row.human_mean, row.human_std)}
      </TableCell>
      <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
        {fmtMeanStd(row.llm_mean, row.llm_std)}
      </TableCell>
    </TableRow>
  );
}

export default function ScoreComparisonTable() {
  const navigate = useNavigate();
  const [data, setData] = useState<HumanAlignmentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHumanAlignment()
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <Typography component="p" sx={{ display: "flex", alignItems: "center", gap: 1, mt: 2 }}>
        <CircularProgress size={20} /> Loading score data…
      </Typography>
    );
  }

  if (error) {
    return <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>;
  }

  if (!data) return null;

  const hasStats = data.dimensions.some((d) => d.human_mean !== null);
  if (!hasStats) return null;

  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom sx={{ mt: 4 }}>
        Score Distribution: Human vs LLM Judge
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Mean score with sample standard deviation in parentheses on the 1–5 Likert scale,
        across {data.n_matched_samples} matched samples.
      </Typography>

      <TableContainer component={Paper} variant="outlined" sx={{ maxWidth: 700 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Dimension</TableCell>
              <TableCell align="right">n</TableCell>
              <Tooltip title="Mean human annotation score (std dev)" placement="top">
                <TableCell align="right">Human</TableCell>
              </Tooltip>
              <Tooltip title="Mean LLM judge score (std dev)" placement="top">
                <TableCell align="right">LLM Judge</TableCell>
              </Tooltip>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.dimensions.map((dim) => (
              <ComparisonRow key={dim.key} row={dim} />
            ))}
            <ComparisonRow row={data.overall} isOverall />
          </TableBody>
        </Table>
      </TableContainer>
      <Button
        variant="outlined"
        size="small"
        sx={{ mt: 1 }}
        onClick={() => navigate("/evaluation-scores")}
      >
        Detail Scores
      </Button>
    </section>
  );
}

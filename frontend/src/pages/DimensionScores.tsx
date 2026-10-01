import { useEffect, useState } from "react";
import { Link as RouterLink, useParams } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import { fetchAnnotatorScores, type AnnotatorScoresData } from "../data";

function getScoreRange(scores: Record<string, number>, annotators: string[]) {
  const values = annotators.map((a) => scores[a]).filter((v) => v != null);
  if (values.length < 2) return { min: null, max: null, divergent: false };
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { min, max, divergent: max - min >= 2 };
}

function scoreCellColor(
  value: number | null | undefined,
  min: number | null,
  max: number | null,
  divergent: boolean,
): string | undefined {
  if (!divergent || value == null || min == null || max == null) return undefined;
  if (value === min) return "error.main";
  if (value === max) return "success.dark";
  return undefined;
}

export default function DimensionScores() {
  const { dimensionKey } = useParams<{ dimensionKey: string }>();
  const [data, setData] = useState<AnnotatorScoresData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnnotatorScores()
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
        <CircularProgress size={20} /> Loading scores…
      </Typography>
    );
  }

  if (error) {
    return <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>;
  }

  if (!data) {
    return <Alert severity="info" sx={{ mt: 2 }}>No annotator score data available.</Alert>;
  }

  const dimension = data.dimensions.find((d) => d.key === dimensionKey);
  if (!dimension) {
    return <Alert severity="warning" sx={{ mt: 2 }}>Dimension "{dimensionKey}" not found.</Alert>;
  }

  const divergentCount = dimension.samples.filter(
    (s) => getScoreRange(s.scores, data.annotators).divergent,
  ).length;

  return (
    <section>
      <Box sx={{ mb: 2 }}>
        <Link component={RouterLink} to="/evaluation-scores" underline="hover" variant="body2">
          ← Evaluation Scores
        </Link>
      </Box>

      <Typography variant="h5" component="h2" gutterBottom>
        {dimension.label}
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Per-sample scores for each annotator on the 1–5 Likert scale. Rows with a ≥2 score gap are
        highlighted; low scores are{" "}
        <Box component="span" sx={{ color: "error.main", fontWeight: 700 }}>
          red
        </Box>{" "}
        and high scores are{" "}
        <Box component="span" sx={{ color: "success.dark", fontWeight: 700 }}>
          green
        </Box>
        .
      </Typography>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Sample ID</TableCell>
              {data.annotators.map((ann) => (
                <TableCell key={ann} align="right" sx={{ whiteSpace: "nowrap" }}>
                  {ann}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {dimension.samples.map((sample) => {
              const { min, max, divergent } = getScoreRange(sample.scores, data.annotators);
              return (
                <TableRow
                  key={sample.sample_id}
                  sx={divergent ? { backgroundColor: (theme) => `${theme.palette.warning.light}22` } : undefined}
                >
                  <TableCell sx={{ fontFamily: "monospace", fontSize: "0.8rem" }}>
                    {sample.sample_id}
                  </TableCell>
                  {data.annotators.map((ann) => {
                    const val = sample.scores[ann];
                    const color = scoreCellColor(val, min, max, divergent);
                    return (
                      <TableCell
                        key={ann}
                        align="right"
                        sx={{
                          fontVariantNumeric: "tabular-nums",
                          color,
                          fontWeight: color ? 700 : undefined,
                        }}
                      >
                        {val != null ? val : "—"}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
            <TableRow sx={{ backgroundColor: "action.hover" }}>
              <TableCell colSpan={data.annotators.length + 1} sx={{ fontWeight: 600 }}>
                Rows with ≥2 score gap:{" "}
                <Box
                  component="span"
                  sx={{ color: divergentCount > 0 ? "warning.dark" : "text.secondary" }}
                >
                  {divergentCount} / {dimension.samples.length}
                </Box>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
    </section>
  );
}

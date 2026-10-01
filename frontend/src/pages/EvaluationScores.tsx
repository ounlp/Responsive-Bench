import { useEffect, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
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

export default function EvaluationScores() {
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
        <CircularProgress size={20} /> Loading annotator scores…
      </Typography>
    );
  }

  if (error) {
    return <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>;
  }

  if (!data) {
    return <Alert severity="info" sx={{ mt: 2 }}>No annotator score data available.</Alert>;
  }

  return (
    <section>
      <Box sx={{ mb: 2 }}>
        <Link component={RouterLink} to="/" underline="hover" variant="body2">
          ← Back
        </Link>
      </Box>

      <Typography variant="h5" component="h2" gutterBottom>
        Evaluation Scores
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Average score per annotator per dimension on the 1–5 Likert scale.
      </Typography>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Dimension</TableCell>
              {data.annotators.map((ann) => (
                <TableCell key={ann} align="right" sx={{ whiteSpace: "nowrap" }}>
                  {ann}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.dimensions.map((dim) => (
              <TableRow key={dim.key}>
                <TableCell>
                  <Link
                    component={RouterLink}
                    to={`/evaluation-scores/${dim.key}`}
                    underline="hover"
                  >
                    {dim.label}
                  </Link>
                </TableCell>
                {data.annotators.map((ann) => (
                  <TableCell
                    key={ann}
                    align="right"
                    sx={{ fontVariantNumeric: "tabular-nums" }}
                  >
                    {dim.annotator_averages[ann] != null
                      ? dim.annotator_averages[ann].toFixed(2)
                      : "—"}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </section>
  );
}

import { useEffect, useState } from "react";
import { Link as RouterLink, useParams } from "react-router-dom";
import Box from "@mui/material/Box";
import Link from "@mui/material/Link";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { fetchJson, type EvalSamplesResponse } from "../data";

export default function EvalSamples() {
  const { runId, evalId } = useParams<{ runId: string; evalId: string }>();
  const [data, setData] = useState<EvalSamplesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId || !evalId) return;
    fetchJson<EvalSamplesResponse>(`runs/${runId}/evals/${evalId}.json`)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [runId, evalId]);

  if (error) return <Typography color="error">{error}</Typography>;
  if (!runId || !evalId || !data) return <Typography>Loading…</Typography>;

  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom>
        {evalId}
      </Typography>
      <Typography paragraph>
        <Link component={RouterLink} to={`/runs/${runId}`} underline="hover">
          ← Back to {runId}
        </Link>
      </Typography>

      <Typography variant="h6" component="h3" gutterBottom>
        Samples
      </Typography>
      <List disablePadding>
        {data.samples.map((s) => (
          <ListItem key={s.id} disablePadding sx={{ mb: 1 }}>
            <Paper variant="outlined" sx={{ width: "100%", p: 1 }}>
              <Box sx={{ flex: 1 }}>
                <Typography fontWeight="bold">Sample {s.id}</Typography>
                <Stack direction="row" spacing={2} sx={{ mt: 0.5 }}>
                  <Link
                    component={RouterLink}
                    to={`/runs/${runId}/evals/${evalId}/samples/${s.id}/results`}
                    underline="hover"
                  >
                    Eval Results
                  </Link>
                </Stack>
              </Box>
            </Paper>
          </ListItem>
        ))}
      </List>
      {data.samples.length === 0 && (
        <Typography color="text.secondary">No samples in this eval.</Typography>
      )}
    </section>
  );
}

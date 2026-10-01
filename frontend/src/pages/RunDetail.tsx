import { useEffect, useState } from "react";
import { Link as RouterLink, useParams } from "react-router-dom";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Link from "@mui/material/Link";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { fetchJson, assetUrl, type RunDetailResponse } from "../data";
import ImageWithFallback from "../ImageWithFallback";

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const [data, setData] = useState<RunDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    fetchJson<RunDetailResponse>(`runs/${runId}.json`)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [runId]);

  if (error) return <Typography color="error">{error}</Typography>;
  if (!runId || !data) return <Typography>Loading…</Typography>;

  return (
    <section>
      <Typography variant="h5" component="h2" gutterBottom>
        {runId}
      </Typography>
      <Typography paragraph>
        <Link component={RouterLink} to="/" underline="hover">
          ← Back to runs
        </Link>
      </Typography>

      <Typography variant="h6" component="h3" gutterBottom>
        Evals
      </Typography>
      <List disablePadding>
        {data.evals.map((e) => (
          <ListItem key={e.eval_id} disablePadding>
            <ListItemButton
              component={RouterLink}
              to={`/runs/${runId}/evals/${e.eval_id}`}
              sx={{ color: "primary.main" }}
            >
              {e.eval_id}
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      {data.evals.length === 0 && (
        <Typography color="text.secondary">No evals for this run.</Typography>
      )}

      <Typography variant="h6" component="h3" sx={{ mt: 3 }} gutterBottom>
        Samples
      </Typography>
      <Stack direction="row" flexWrap="wrap" useFlexGap spacing={2}>
        {data.samples.map((s) => (
          <Card key={s.id} sx={{ maxWidth: 320 }}>
            <CardContent>
              <Typography fontWeight="bold" gutterBottom>
                Sample {s.id}
              </Typography>
              {(s.sketch_desktop_path ?? s.sketch_mobile_path) && (
                <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                  {s.sketch_desktop_path && (
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                        Sketch (Desktop)
                      </Typography>
                      <ImageWithFallback
                        src={assetUrl(s.sketch_desktop_path)}
                        alt={`Sketch Desktop ${s.id}`}
                        fallbackLabel="Image not found"
                        style={{ width: "100%", height: "auto" }}
                      />
                    </Box>
                  )}
                  {s.sketch_mobile_path && (
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                        Sketch (Mobile)
                      </Typography>
                      <ImageWithFallback
                        src={assetUrl(s.sketch_mobile_path)}
                        alt={`Sketch Mobile ${s.id}`}
                        fallbackLabel="Image not found"
                        style={{ width: "100%", height: "auto" }}
                      />
                    </Box>
                  )}
                </Stack>
              )}
              {(s.reference_path ?? s.generated_path) && (
                <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                  {s.reference_path && (
                    <Box sx={{ flex: 1 }}>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        display="block"
                        gutterBottom
                      >
                        Reference
                      </Typography>
                      <ImageWithFallback
                        src={assetUrl(s.reference_path)}
                        alt={`Reference ${s.id}`}
                        fallbackLabel="Image not found"
                        style={{ width: "100%", height: "auto" }}
                      />
                    </Box>
                  )}
                  {s.generated_path && (
                    <Box sx={{ flex: 1 }}>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        display="block"
                        gutterBottom
                      >
                        Generated
                      </Typography>
                      <ImageWithFallback
                        src={assetUrl(s.generated_path)}
                        alt={`Generated ${s.id}`}
                        fallbackLabel="Image not found"
                        style={{ width: "100%", height: "auto" }}
                      />
                    </Box>
                  )}
                </Stack>
              )}
            </CardContent>
          </Card>
        ))}
      </Stack>
    </section>
  );
}

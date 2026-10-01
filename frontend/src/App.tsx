import { Routes, Route, Link as RouterLink } from "react-router-dom";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import Link from "@mui/material/Link";
import Typography from "@mui/material/Typography";
import RunsList from "./pages/RunsList";
import RunDetail from "./pages/RunDetail";
import EvalSamples from "./pages/EvalSamples";
import EvalResults from "./pages/EvalResults";
import EvaluationScores from "./pages/EvaluationScores";
import DimensionScores from "./pages/DimensionScores";
import AlignmentComparison from "./pages/AlignmentComparison";
import { LEADERBOARD_ONLY } from "./data";

function App() {
  return (
    <Container maxWidth={false} disableGutters>
      <Box sx={{ py: 2, px: { xs: 2, sm: 3, md: 4 } }}>
        <Typography variant="h4" component="h1" gutterBottom>
          <Link component={RouterLink} to="/" color="inherit" underline="hover">
            ResponsiveBench Runs Explorer
          </Link>
        </Typography>
        <Routes>
        <Route path="/" element={<RunsList />} />
        {!LEADERBOARD_ONLY && (
          <>
            <Route path="/runs/:runId" element={<RunDetail />} />
            <Route path="/runs/:runId/evals/:evalId" element={<EvalSamples />} />
            <Route
              path="/runs/:runId/evals/:evalId/samples/:sampleId/results"
              element={<EvalResults />}
            />
              </>
        )}
        <Route path="/evaluation-scores" element={<EvaluationScores />} />
        <Route path="/evaluation-scores/:dimensionKey" element={<DimensionScores />} />
        <Route path="/alignment-comparison" element={<AlignmentComparison />} />
      </Routes>
      </Box>
    </Container>
  );
}

export default App;

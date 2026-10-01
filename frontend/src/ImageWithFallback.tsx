import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

interface ImageWithFallbackProps {
  src: string | null;
  alt: string;
  fallbackLabel?: string;
  style?: React.CSSProperties;
}

function Placeholder({ label }: { label: string }) {
  return (
    <Box
      sx={{
        minHeight: 80,
        bgcolor: "action.hover",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
    </Box>
  );
}

export default function ImageWithFallback({
  src,
  alt,
  fallbackLabel = "Image not found",
  style,
}: ImageWithFallbackProps) {
  const [errored, setErrored] = useState(false);

  if (!src) return <Placeholder label="No image" />;
  if (errored) return <Placeholder label={fallbackLabel} />;

  return (
    <img
      src={src}
      alt={alt}
      style={style}
      onError={() => setErrored(true)}
    />
  );
}

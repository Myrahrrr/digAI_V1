import { Box, Button, Container, Divider, Typography } from "@mui/material";
import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

export default function End() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const prolific = localStorage.getItem("prolific");

  const restart = () => {
    localStorage.clear();
    navigate("/");
  };
  useEffect(() => {}, []);
  return (
    <Box
      className="background"
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
      }}
    >
      <Container>
        <Box component="main" sx={{ textAlign: "center" }}>
          <Typography variant="h4" sx={{ color: "#ffffff" }}>
            {t("end.title")}
          </Typography>
          <Divider sx={{ margin: 2 }} />
          <Typography sx={{ color: "#fafafa" }}>{t("end.p1")}</Typography>
          {prolific && (
            <Typography sx={{ color: "#fafafa" }}>
              {t("end.prolific")}
            </Typography>
          )}
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              marginTop: 2,
              gap: 2,
            }}
          >
            <Button variant="contained" onClick={restart}>
              {t("end.restart")}
            </Button>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}

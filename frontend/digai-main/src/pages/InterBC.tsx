import { Box, Button, Container, Divider, Typography } from "@mui/material";
import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import ChatbotService from "../services/chatbotService";
import { useTranslation } from "react-i18next";

export default function InterBC() {
  const { t } = useTranslation();
  let uuid = useRef<string | null>(null);
  const navigate = useNavigate();

  const startPartC = async () => {
    const session = await ChatbotService.navigate(uuid.current!, "/part-c");
    localStorage.setItem("state", session.state);
    navigate("/part-c");
  };

  useEffect(() => {
    uuid.current = localStorage.getItem("uuid");
    if (!uuid) {
      navigate("/");
    }
  }, []);
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
            {t("inter_bc.title")}
          </Typography>
          <Divider sx={{ margin: 2 }} />
          <Typography sx={{ color: "#fafafa" }}>{t("inter_bc.p1")}</Typography>
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              marginTop: 2,
              gap: 2,
            }}
          >
            <Button variant="contained" onClick={startPartC}>
              {t("inter_bc.start")}
            </Button>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}

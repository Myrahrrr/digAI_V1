import { Box, Button, Container, Divider, Typography } from "@mui/material";
import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import ChatbotService from "../services/chatbotService";
import { useTranslation } from "react-i18next";

export default function InterAB() {
  const { t } = useTranslation();
  let uuid = useRef<string | null>(null);
  const navigate = useNavigate();

  const startPartB = async () => {
    const session = await ChatbotService.navigate(uuid.current!, "/part-b");
    localStorage.setItem("state", session.state);
    navigate("/part-b");
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
            {t("inter_ab.title")}
          </Typography>
          <Divider sx={{ margin: 2 }} />
          <Typography sx={{ color: "#fafafa" }}>{t("inter_ab.p1")}</Typography>
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              marginTop: 2,
              gap: 2,
            }}
          >
            <Button variant="contained" onClick={startPartB}>
              {t("inter_ab.start")}
            </Button>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}

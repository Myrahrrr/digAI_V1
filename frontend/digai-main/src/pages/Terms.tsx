import { Box, Container, Divider, Typography } from "@mui/material";
import { useTranslation } from "react-i18next";

export default function Terms() {
  const { t } = useTranslation();

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
          <Box sx={{ display: "flex", flexDirection: "row" }}>
            <img
              src="https://static.cct.ufcg.edu.br/logo/UFCG.svg"
              height={150}
            />
            <Typography variant="h4" sx={{ color: "#ffffff" }}>
              {t("title")}
            </Typography>
            <img
              src="https://static.cct.ufcg.edu.br/logo/UFAL.svg"
              height={150}
            />
          </Box>
          <Divider sx={{ margin: 2 }} />
          <Typography variant="h5" sx={{ color: "#ffffff" }}>
            {t("terms.title1")}
          </Typography>
          <Typography sx={{ color: "#fafafa" }}>{t("terms.p1")}</Typography>
          <Typography variant="h5" sx={{ color: "#ffffff", mt: 2 }}>
            {t("terms.title2")}
          </Typography>
          <Typography sx={{ color: "#fafafa" }}>{t("terms.p2")}</Typography>

          <Typography variant="h5" sx={{ color: "#ffffff", mt: 2 }}>
            {t("terms.title3")}
          </Typography>
          <Typography sx={{ color: "#fafafa" }}>{t("terms.p3")}</Typography>

          <Typography variant="h5" sx={{ color: "#ffffff", mt: 2 }}>
            {t("terms.title4")}
          </Typography>
          <Typography sx={{ color: "#fafafa" }}>{t("terms.p4")}</Typography>
        </Box>
      </Container>
    </Box>
  );
}

import { Box, Button, Container, Grid, Typography } from "@mui/material";
import ClothItem from "../components/ClothItem";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import RecommenderService from "../services/recommenderService";
import ChatbotService from "../services/chatbotService";
import PromptDialog from "../components/PromptDialog";
import { useTranslation } from "react-i18next";

export default function PartA() {
  const { t } = useTranslation();
  let uuid = useRef<string | null>(null);
  let productSelected = useRef<string | null>(null);
  const moreRecommendationsRef = useRef<HTMLDivElement>();
  let [state, setState] = useState<any>(null);
  const [items, setItems] = useState<string[]>([]);
  const [dialog, setDialog] = useState({
    open: false,
    title: "",
    description: "",
  });
  const navigate = useNavigate();

  const finishExperiment = () => {
    ChatbotService.finish(uuid.current!, productSelected.current!).then(
      (session) => {
        localStorage.setItem("state", JSON.stringify(session.state));
        navigate(session.state.page);
      }
    );
  };

  useEffect(() => {
    let ignore = false;
    uuid.current = localStorage.getItem("uuid");
    const state = JSON.parse(localStorage.getItem("state") || "null");
    if (!uuid.current || !state) {
      navigate("/");
    } else {
      setState((_: any) => state);
      const lastItems = localStorage.getItem("itemsA");
      if (lastItems) {
        setItems((_) => JSON.parse(lastItems));
      } else {
        RecommenderService.recomm(state).then((newItems) => {
          if (!ignore) {
            setItems((at) => {
              const newList = at.concat(newItems.products);
              localStorage.setItem("itemsA", JSON.stringify(newList));
              return newList;
            });
          }
        });
      }
    }
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    moreRecommendationsRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items]);

  return (
    <Box component="main">
      <Box component="header" className="header" sx={{ textAlign: "center" }}>
        <Container sx={{ padding: 4 }}>
          <Typography variant="h5">{t("title2")}</Typography>
        </Container>
      </Box>
      <Container>
        <Grid container padding={2} spacing={2}>
          {items.map((item, index) => (
            <Grid item key={index} xs={12} md={3}>
              <ClothItem
                onLike={(liked) => {
                  if (liked) {
                    ChatbotService.like(uuid.current!, item).then((session) => {
                      setState((_: any) => {
                        localStorage.setItem(
                          "state",
                          JSON.stringify(session.state)
                        );
                        return session.state;
                      });
                    });
                  } else {
                    ChatbotService.dislike(uuid.current!, item).then(
                      (session) => {
                        setState((_: any) => {
                          localStorage.setItem(
                            "state",
                            JSON.stringify(session.state)
                          );
                          return session.state;
                        });
                      }
                    );
                  }
                }}
                onShop={() => {
                  productSelected.current = item;
                  setDialog({
                    open: true,
                    title: t("dialog.finish.title"),
                    description: t("dialog.finish.description"),
                  });
                }}
                id={item}
              />
            </Grid>
          ))}
        </Grid>
        <Box
          sx={{ textAlign: "center", margin: 2 }}
          ref={moreRecommendationsRef}
        >
          <Button
            variant="contained"
            onClick={() => {
              RecommenderService.recomm(state).then((newItems) => {
                setItems((at) => {
                  const newList = at.concat(newItems.products);
                  localStorage.setItem("itemsA", JSON.stringify(newList));
                  return newList;
                });
              });
            }}
          >
            {t("recommend")}
          </Button>
        </Box>
        <PromptDialog
          open={dialog.open}
          title={dialog.title}
          description={dialog.description}
          onClose={() => {
            setDialog({
              open: false,
              title: "",
              description: "",
            });
          }}
          onOk={finishExperiment}
        />
      </Container>
    </Box>
  );
}

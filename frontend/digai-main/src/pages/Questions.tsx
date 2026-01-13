import {
  Alert,
  Box,
  Button,
  Container,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from "@mui/material";
import { createRef, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import ChatbotService from "../services/chatbotService";
import Likert from "../components/forms/Likert";
import Text from "../components/forms/Text";
import Adjectives from "../components/forms/Adjectives";
import PromptDialog from "../components/PromptDialog";
import { useTranslation } from "react-i18next";

export type QuestionItem = {
  start: string;
  end: string;
};

export type Question = {
  id: string;
  type: string;
  title: string;
  required?: boolean;
  items?: QuestionItem[];
};

export type Section = {
  title: string;
  questions: Question[];
};

export default function Questions() {
  const { t } = useTranslation();
  const uuid = useRef<string | null>(null);
  const topAnchor = createRef<HTMLDivElement>();
  const [data, setData] = useState<any>({});
  const [err, setErr] = useState<string>("");
  const [activeStep, setActiveStep] = useState(0);
  const [dialog, setDialog] = useState({
    open: false,
    title: "",
    description: "",
  });
  const navigate = useNavigate();
  const sections: Section[] = [
    {
      title: t("qst.sec1.title"),
      questions: [
        {
          id: "q1",
          type: "likert",
          title: "Q1: " + t("qst.sec1.q1"),
          required: true,
        },
        {
          id: "q1c",
          type: "text",
          title: "Q1: " + t("qst.comments1"),
        },
        {
          id: "q2",
          type: "likert",
          title: "Q2: " + t("qst.sec1.q2"),
          required: true,
        },
        {
          id: "q2c",
          type: "text",
          title: "Q2: " + t("qst.comments2"),
        },
        {
          id: "q3",
          type: "likert",
          title: "Q3: " + t("qst.sec1.q3"),
          required: true,
        },
        {
          id: "q3c",
          type: "text",
          title: "Q3: " + t("qst.comments2"),
        },
      ],
    },
    {
      title: t("qst.sec2.title"),
      questions: [
        {
          id: "d1",
          type: "likert",
          title: "D1: " + t("qst.sec2.q1"),
          required: true,
        },
        {
          id: "d1c",
          type: "text",
          title: "D1: " + t("qst.comments2"),
        },
        {
          id: "d2",
          type: "likert",
          title: "D2: " + t("qst.sec2.q2"),
          required: true,
        },
        {
          id: "d2c",
          type: "text",
          title: "D2: " + t("qst.comments2"),
        },
        {
          id: "d3",
          type: "likert",
          title: "D3: " + t("qst.sec2.q3"),
          required: true,
        },
        {
          id: "d3c",
          type: "text",
          title: "D3: " + t("qst.comments2"),
        },
      ],
    },
    {
      title: t("qst.sec3.title"),
      questions: [
        {
          id: "c1",
          type: "likert",
          title: "C1: " + t("qst.sec3.q1"),
          required: true,
        },
        {
          id: "c1c",
          type: "text",
          title: "C1: " + t("qst.comments2"),
        },
        {
          id: "c2",
          type: "likert",
          title: "C2: " + t("qst.sec3.q2"),
          required: true,
        },
        {
          id: "c2c",
          type: "text",
          title: "C2: " + t("qst.comments2"),
        },
        {
          id: "c3",
          type: "likert",
          title: "C3: " + t("qst.sec3.q3"),
          required: true,
        },
        {
          id: "c3c",
          type: "text",
          title: "C3: " + t("qst.comments2"),
        },
        {
          id: "c4",
          type: "likert",
          title: "C4: " + t("qst.sec3.q4"),
          required: true,
        },
        {
          id: "c4c",
          type: "text",
          title: "C4: " + t("qst.comments2"),
        },
      ],
    },
    {
      title: t("qst.sec4.title"),
      questions: [
        {
          id: "e1",
          type: "likert",
          title: "E1: " + t("qst.sec4.q1"),
          required: true,
        },
        {
          id: "e1c",
          type: "text",
          title: "E1: " + t("qst.comments2"),
        },
        {
          id: "e2",
          type: "likert",
          title: "E2: " + t("qst.sec4.q2"),
          required: true,
        },
        {
          id: "e2c",
          type: "text",
          title: "E2: " + t("qst.comments2"),
        },
        {
          id: "e3",
          type: "likert",
          title: "E3: " + t("qst.sec4.q3"),
          required: true,
        },
        {
          id: "e3c",
          type: "text",
          title: "E3: " + t("qst.comments2"),
        },
      ],
    },
    {
      title: t("qst.sec5.title"),
      questions: [
        {
          id: "t1",
          type: "likert",
          title: "T1: " + t("qst.sec5.q1"),
          required: true,
        },
        {
          id: "t1c",
          type: "text",
          title: "T1: " + t("qst.comments2"),
        },
        {
          id: "t2",
          type: "likert",
          title: "T2: " + t("qst.sec5.q2"),
          required: true,
        },
        {
          id: "t2c",
          type: "text",
          title: "T2: " + t("qst.comments2"),
        },
        {
          id: "t3",
          type: "likert",
          title: "T3: " + t("qst.sec5.q3"),
          required: true,
        },
        {
          id: "t3c",
          type: "text",
          title: "T3: " + t("qst.comments2"),
        },
        {
          id: "t4",
          type: "likert",
          title: "T4: " + t("qst.sec5.q4"),
          required: true,
        },
        {
          id: "t4c",
          type: "text",
          title: "T4: " + t("qst.comments2"),
        },
      ],
    },
    {
      title: t("qst.sec6.title"),
      questions: [
        {
          id: "s1",
          type: "likert",
          title: "S1: " + t("qst.sec6.q1"),
          required: true,
        },
        {
          id: "s1c",
          type: "text",
          title: "S1: " + t("qst.comments2"),
        },
        {
          id: "s2",
          type: "likert",
          title: "S2: " + t("qst.sec6.q2"),
          required: true,
        },
        {
          id: "s2c",
          type: "text",
          title: "S2: " + t("qst.comments2"),
        },
        {
          id: "s3",
          type: "likert",
          title: "S3: " + t("qst.sec6.q3"),
          required: true,
        },
        {
          id: "s3c",
          type: "text",
          title: "S3: " + t("qst.comments2"),
        },
        {
          id: "s4",
          type: "likert",
          title: "S4: " + t("qst.sec6.q4"),
          required: true,
        },
        {
          id: "s4c",
          type: "text",
          title: "S4: " + t("qst.comments2"),
        },
      ],
    },
    {
      title: t("qst.sec7.title"),
      questions: [
        {
          id: "a1",
          type: "adjectives",
          title: t("qst.sec7.q1"),
          required: true,
          items: [
            { start: t("adjectives.boring"), end: t("adjectives.motivating") },
            { start: t("adjectives.practical"), end: t("adjectives.pleasant") },
            {
              start: t("adjectives.misleading"),
              end: t("adjectives.trustworthy"),
            },
            { start: t("adjectives.isolating"), end: t("adjectives.sociable") },
            {
              start: t("adjectives.machinelike"),
              end: t("adjectives.humanlike"),
            },
            {
              start: t("adjectives.artificial"),
              end: t("adjectives.lifelike"),
            },
            { start: t("adjectives.fake"), end: t("adjectives.natural") },
          ],
        },
        {
          id: "a2",
          type: "adjectives",
          title: t("qst.sec7.q2"),
          required: true,
          items: [
            { start: t("adjectives.boring"), end: t("adjectives.motivating") },
            { start: t("adjectives.practical"), end: t("adjectives.pleasant") },
            {
              start: t("adjectives.misleading"),
              end: t("adjectives.trustworthy"),
            },
            { start: t("adjectives.isolating"), end: t("adjectives.sociable") },
            {
              start: t("adjectives.machinelike"),
              end: t("adjectives.humanlike"),
            },
            {
              start: t("adjectives.artificial"),
              end: t("adjectives.lifelike"),
            },
            { start: t("adjectives.fake"), end: t("adjectives.natural") },
          ],
        },
      ],
    },
  ];

  const isEmpty = (value: any): boolean => {
    return value === "" || value === 0 || value === null || value === undefined;
  };

  const isInvalid = (question: Question) => {
    switch (question.type) {
      case "likert":
        return (
          question.required === true &&
          (isEmpty(data?.[question.id]?.a) || isEmpty(data?.[question.id]?.b))
        );
      case "text":
        return question.required === true && isEmpty(data?.[question.id]);
      case "adjectives":
        return (
          question.required === true &&
          (data?.[question.id]?.length === 0 ||
            data?.[question.id]?.length !== question.items!.length ||
            data?.[question.id]?.some((itemValue: string) =>
              isEmpty(itemValue)
            ))
        );
    }
    return true;
  };

  const handleNext = async () => {
    const invalid = sections[activeStep].questions.some((question) =>
      isInvalid(question)
    );
    if (invalid) {
      setErr(t("qst.error.required"));
      return;
    }
    if (activeStep === sections.length - 1) {
      setDialog({
        open: true,
        title: t("dialog.qst.title"),
        description: t("dialog.qst.description"),
      });
    } else {
      await ChatbotService.questionnaire(uuid.current!, data, false);
      setActiveStep((lastStap) => lastStap + 1);
      setErr((_) => "");
      topAnchor.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  const submitQuestionnaire = async () => {
    await ChatbotService.questionnaire(uuid.current!, data, true);
    const session = await ChatbotService.navigate(uuid.current!, "/inter-bc");
    localStorage.setItem("state", session.state);
    navigate("/inter-bc");
  };

  const handleChange = (id: any, value: any) => {
    setData((lastData: any) => {
      const newData = { ...lastData, [id]: value };
      localStorage.setItem("data", JSON.stringify(newData));
      return newData;
    });
  };

  useEffect(() => {
    uuid.current = localStorage.getItem("uuid");
    if (!uuid.current) {
      navigate("/");
    }
    const lastData = localStorage.getItem("data");
    if (lastData) {
      setData((_: any) => JSON.parse(lastData));
    }
  }, []);

  return (
    <Box ref={topAnchor} component="main">
      <Box component="header" className="header" sx={{ textAlign: "center" }}>
        <Container>
          <Box
            sx={{
              display: "flex",
              flexDirection: "row",
              justifyContent: "center",
              padding: 4,
            }}
          >
            <img
              src="https://static.cct.ufcg.edu.br/logo/UFCG.svg"
              width={70}
            />
            <Typography variant="h5" sx={{ margin: 4 }}>
              Questionnaire
            </Typography>
            <img
              src="https://static.cct.ufcg.edu.br/logo/UFAL.svg"
              width={50}
            />
          </Box>
        </Container>
      </Box>
      <Container>
        <Box sx={{ margin: 4 }}>
          <Stepper activeStep={activeStep}>
            {sections.map((_, index) => (
              <Step key={index} completed={activeStep > index}>
                <StepLabel />
              </Step>
            ))}
          </Stepper>
        </Box>
        <Box sx={{ margin: 4 }}>
          <Typography variant="h4">{sections[activeStep].title}</Typography>
          {sections[activeStep].questions.map((question, index) => {
            switch (question.type) {
              case "likert":
                return (
                  <Likert
                    key={index}
                    id={question.id}
                    title={question.title}
                    required={question.required}
                    value={data[question.id]}
                    onChange={handleChange}
                    invalid={
                      err !== "" &&
                      question.required === true &&
                      isInvalid(question)
                    }
                  />
                );
              case "text":
                return (
                  <Text
                    key={index}
                    id={question.id}
                    title={question.title}
                    required={question.required}
                    value={data[question.id]}
                    onChange={handleChange}
                    invalid={
                      err !== "" &&
                      question.required === true &&
                      isInvalid(question)
                    }
                  />
                );
              case "adjectives":
                return (
                  <Adjectives
                    key={index}
                    id={question.id}
                    title={question.title}
                    items={question.items!}
                    required={question.required}
                    value={data[question.id]}
                    onChange={handleChange}
                    invalid={
                      err !== "" &&
                      question.required === true &&
                      isInvalid(question)
                    }
                  />
                );
              default:
                console.log(`${question.type} not implemented`);
            }
          })}
        </Box>
        <Box sx={{ textAlign: "center", margin: 2 }}>
          {err && (
            <Alert sx={{ margin: 2 }} severity="error">
              {err}
            </Alert>
          )}
          <Button variant="contained" onClick={handleNext}>
            {activeStep === sections.length - 1 ? t('finish') : t('next')}
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
          onOk={submitQuestionnaire}
        />
      </Container>
    </Box>
  );
}

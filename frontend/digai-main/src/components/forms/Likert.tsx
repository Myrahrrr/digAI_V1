import {
  Card,
  CardContent,
  CardHeader,
  Paper,
  Radio,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  styled,
  tableCellClasses,
} from "@mui/material";
import { grey, red } from "@mui/material/colors";
import { t } from "i18next";

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: grey[400],
    color: theme.palette.common.black,
  },
  [`&.${tableCellClasses.body}`]: {
    color: theme.palette.common.black,
  },
}));
const StyledTableRow = styled(TableRow)(({ theme }) => ({
  "&:nth-of-type(odd)": {
    backgroundColor: theme.palette.action.hover,
  },
  // hide last border
  "&:last-child td, &:last-child th": {
    border: 0,
  },
}));

export type LikertProps = {
  id: string;
  title: string;
  required?: boolean;
  value: { a: string; b: string };
  onChange: (id: string, value: { a: string; b: string }) => void;
  invalid: boolean;
};

export default function Likert({
  id,
  title,
  required,
  value,
  onChange,
  invalid,
}: LikertProps) {
  const handleA = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newData = { ...value, a: event.target.value };
    onChange(id, newData);
  };
  const handleB = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newData = { ...value, b: event.target.value };
    onChange(id, newData);
  };
  return (
    <Card sx={{ margin: 2 }}>
      <CardHeader
        sx={{ color: invalid ? red[400] : "#000000" }}
        title={title + (required ? "*" : "")}
      />
      <CardContent>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("part")}
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("qst.likert.l1")}
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("qst.likert.l2")}
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("qst.likert.l3")}
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("qst.likert.l4")}
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("qst.likert.l5")}
                </StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              <StyledTableRow>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("part")} A
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.a === "1"}
                    onChange={handleA}
                    value="1"
                    name="partA"
                    inputProps={{ "aria-label": "1" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.a === "2"}
                    onChange={handleA}
                    value="2"
                    name="partA"
                    inputProps={{ "aria-label": "2" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.a === "3"}
                    onChange={handleA}
                    value="3"
                    name="partA"
                    inputProps={{ "aria-label": "3" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.a === "4"}
                    onChange={handleA}
                    value="4"
                    name="partA"
                    inputProps={{ "aria-label": "4" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.a === "5"}
                    onChange={handleA}
                    value="5"
                    name="partA"
                    inputProps={{ "aria-label": "5" }}
                  />
                </StyledTableCell>
              </StyledTableRow>
              <StyledTableRow>
                <StyledTableCell sx={{ width: "16%" }}>
                  {t("part")} B
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.b === "1"}
                    onChange={handleB}
                    value="1"
                    name="partB"
                    inputProps={{ "aria-label": "1" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.b === "2"}
                    onChange={handleB}
                    value="2"
                    name="partB"
                    inputProps={{ "aria-label": "2" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.b === "3"}
                    onChange={handleB}
                    value="3"
                    name="partB"
                    inputProps={{ "aria-label": "3" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.b === "4"}
                    onChange={handleB}
                    value="4"
                    name="partB"
                    inputProps={{ "aria-label": "4" }}
                  />
                </StyledTableCell>
                <StyledTableCell sx={{ width: "16%" }}>
                  <Radio
                    checked={value?.b === "5"}
                    onChange={handleB}
                    value="5"
                    name="partB"
                    inputProps={{ "aria-label": "5" }}
                  />
                </StyledTableCell>
              </StyledTableRow>
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
}

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

export type AdjectivesProps = {
  id: string;
  title: string;
  items: { start: string; end: string }[];
  required?: boolean;
  value: string[];
  onChange: (id: string, value: string[]) => void;
  invalid: boolean;
};

export default function Adjectives({
  id,
  title,
  items,
  required,
  value,
  onChange,
  invalid,
}: AdjectivesProps) {
  const handleChange = (index: number) => {
    return (event: React.ChangeEvent<HTMLInputElement>) => {
      const newData = [...(value || [])];
      newData[index] = event.target.value;
      onChange(id, newData);
    };
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
                <StyledTableCell sx={{ width: "14%" }}></StyledTableCell>
                <StyledTableCell sx={{ width: "14%" }}>1</StyledTableCell>
                <StyledTableCell sx={{ width: "14%" }}>2</StyledTableCell>
                <StyledTableCell sx={{ width: "14%" }}>3</StyledTableCell>
                <StyledTableCell sx={{ width: "14%" }}>4</StyledTableCell>
                <StyledTableCell sx={{ width: "14%" }}>5</StyledTableCell>
                <StyledTableCell sx={{ width: "14%" }}></StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {items.map((item, index) => (
                <StyledTableRow>
                  <StyledTableCell sx={{ width: "14%" }}>
                    {item.start}
                  </StyledTableCell>
                  <StyledTableCell sx={{ width: "14%" }}>
                    <Radio
                      checked={value?.[index] === "1"}
                      onChange={handleChange(index)}
                      value="1"
                      name={id + "-" + index}
                      inputProps={{ "aria-label": "1" }}
                    />
                  </StyledTableCell>
                  <StyledTableCell sx={{ width: "14%" }}>
                    <Radio
                      checked={value?.[index] === "2"}
                      onChange={handleChange(index)}
                      value="2"
                      name={id + "-" + index}
                      inputProps={{ "aria-label": "2" }}
                    />
                  </StyledTableCell>
                  <StyledTableCell sx={{ width: "14%" }}>
                    <Radio
                      checked={value?.[index] === "3"}
                      onChange={handleChange(index)}
                      value="3"
                      name={id + "-" + index}
                      inputProps={{ "aria-label": "3" }}
                    />
                  </StyledTableCell>
                  <StyledTableCell sx={{ width: "14%" }}>
                    <Radio
                      checked={value?.[index] === "4"}
                      onChange={handleChange(index)}
                      value="4"
                      name={id + "-" + index}
                      inputProps={{ "aria-label": "4" }}
                    />
                  </StyledTableCell>
                  <StyledTableCell sx={{ width: "14%" }}>
                    <Radio
                      checked={value?.[index] === "5"}
                      onChange={handleChange(index)}
                      value="5"
                      name={id + "-" + index}
                      inputProps={{ "aria-label": "5" }}
                    />
                  </StyledTableCell>
                  <StyledTableCell sx={{ width: "14%" }}>
                    {item.end}
                  </StyledTableCell>
                </StyledTableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
}

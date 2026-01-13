import { Card, CardContent, CardHeader, TextField } from "@mui/material";
import { red } from "@mui/material/colors";

export type TextProps = {
  id: string;
  title: string;
  required?: boolean;
  value: string;
  onChange: (id: string, value: string) => void;
  invalid: boolean;
};

export default function Text({
  id,
  title,
  required,
  value,
  onChange,
  invalid,
}: TextProps) {
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(id, event.target.value);
  };
  return (
    <Card sx={{ margin: 2 }}>
      <CardHeader
        sx={{ color: invalid ? red[400] : "#000000" }}
        title={title + (required ? "*" : "")}
      />
      <CardContent>
        <TextField
          id="filled-multiline-static"
          multiline
          fullWidth
          rows={4}
          variant="filled"
          value={value || ""}
          onChange={handleChange}
        />
      </CardContent>
    </Card>
  );
}

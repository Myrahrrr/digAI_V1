import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@mui/material";
import { useTranslation } from "react-i18next";

export type PromptDialogProps = {
  title: string;
  description: string;
  open: boolean;
  onOk: () => void;
  onClose: () => void;
};

export default function PromptDialog(props: PromptDialogProps) {
  const { t } = useTranslation();
  return (
    <Dialog
      open={props.open}
      onClose={props.onClose}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title">{props.title}</DialogTitle>
      <DialogContent>
        <DialogContentText id="alert-dialog-description">
          {props.description}
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button autoFocus onClick={props.onClose}>
          {t("cancel")}
        </Button>
        <Button onClick={props.onOk}>Ok</Button>
      </DialogActions>
    </Dialog>
  );
}

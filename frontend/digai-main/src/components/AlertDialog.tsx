import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@mui/material";

export type AlertDialogProps = {
  title: string;
  description: string;
  open: boolean;
  onClose: () => void;
};

export default function AlertDialog(props: AlertDialogProps) {
  return (
    <Dialog
      open={props.open}
      onClose={props.onClose}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title">
        {props.title}
      </DialogTitle>
      <DialogContent>
        <DialogContentText id="alert-dialog-description">
          {props.description}
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={props.onClose} autoFocus>
          OK
        </Button>
      </DialogActions>
    </Dialog>
  );
}
